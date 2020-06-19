import numpy as np
import torch
import pytorch3d
from pytorch3d.structures import Meshes, Textures, join_meshes_as_batch
from pytorch3d.io import load_obj
from pytorch3d.loss.mesh_laplacian_smoothing import laplacian_cot
import networkx as nx
from pytorch3d.structures import utils as struct_utils
from collections import defaultdict
from time import perf_counter
from scipy.optimize import lsq_linear
from utils import least_sq_with_known_values, Timer
from tqdm import tqdm
from matplotlib import pyplot as plt

def ARAP_from_meshes(meshes):
	"""Produce ARAPMeshes object from Meshes object
	:type meshes: Meshes
	"""
	verts = meshes.verts_list()
	faces = meshes.faces_list()
	textures = meshes.textures

	return ARAPMeshes(verts=verts, faces=faces, textures=textures)

def add_one_ring_neighbours(meshes, arr, mesh_idx=0):
	"""Given an array of mesh vertex indices, returns an array of all arr, plus any neighbouring verts
	:type meshes: ARAPMeshes
	:type arr: list
	"""
	arr = set(arr)
	for v in list(arr):
		for n in meshes.one_ring_neighbours[mesh_idx][v]:
			arr.add(n)

	return list(arr)

def add_n_ring_neighbours(meshes, arr, mesh_idx=0, n=2):
	"""Recursively retrieve n ring neighbours for array"""
	for i in range(n):
		arr = add_one_ring_neighbours(meshes, arr, mesh_idx=mesh_idx)

	return arr

def cholesky_invert(tensor):

	L_L = torch.cholesky(tensor)
	L_L_inv = torch.inverse(L_L)
	L_ch_inv = torch.mm(L_L_inv.T, L_L_inv)

	return L_ch_inv

class ARAPMeshes(Meshes):
	"""
	Subclass of PyTorch3D Meshes.
	Upon deformation, provided with 'static' vertices, and 'handle' vertices.
	Allows for movement of handle vertices from original mesh position, and identifies new positions of all other
	non-static verts using the As Rigid As Possible algorithm (As-Rigid-As-Possible Surface Modeling, O. Sorkine & M. Alexa)"""

	def __init__(self, verts=None, faces=None, textures=None):
		"""
		lists of verts, faces and textures for methods. For details, see Meshes documentation
		"""

		super().__init__(verts, faces, textures)

		self.one_ring_neighbours = self.get_one_ring_neighbours()

		self.L, self.w, self.L_inv = self.precompute()  # Precomputed edge weights and Laplace-Beltrami operator

		self.C = self.verts_padded().mean(dim=1) # centre of mass for each mesh

		self.timer = Timer()

	def get_one_ring_neighbours(self):
		"""Return a dict, where key i gives a list of all neighbouring vertices (connected by exactly one edge)"""

		all_faces = self.faces_padded().detach().numpy()
		orn = []

		for m, faces in enumerate(all_faces): # for each mesh
			mapping = defaultdict(set)  # start with sets, to prevent repeats
			for f in faces:
				for j in [0,1,2]: # for each vert in face
					i, k = (j+1)%3, (j+2)%3  # get 2 other vertices
					mapping[f[j]].add(f[i])
					mapping[f[j]].add(f[k])

			mapping_out = {k: list(v) for k, v in mapping.items()} # convert to list for easier torch/numpy integration
			orn.append(mapping_out)

		return orn

	def precompute(self):
		"""Precompute edge weights and Laplacian-Beltrami operator"""

		n = self.num_verts_per_mesh()
		p = self.verts_padded()  # undeformed mesh

		# pre-factored edge weightings, and Laplace-Beltrami operator
		w_packed, _ = laplacian_cot(self)  # per-edge weightings used
		w_packed = 0.5 * w_packed.to_dense()

		# diagonal L_ii = sum(w_i). non-diagonal L_ij = -w_ij
		L_packed = torch.diag(torch.sum(w_packed, dim=1)) - w_packed

		## w is packed currently. Convert to padded
		max_n = max(n.tolist())
		w_padded = torch.zeros((len(self), max_n, max_n))
		L_padded = torch.zeros_like(w_padded)
		L_inv_padded = torch.zeros_like(w_padded)

		# extract w for each mesh
		cum_n = np.concatenate([[0], np.cumsum(n.tolist())])  # list of n_verts before and after every mesh is included

		for i, (start, end) in enumerate(zip(cum_n[:-1], cum_n[1:])):
			w_padded[i] = w_packed[start:end, start:end]
			L_padded[i] = L_packed[start:end, start:end]
			# L_inv_padded[i] = torch.inverse(L_padded[i])
			#

			L_inv_padded[i] = cholesky_invert(L_padded[i])

		return L_padded, w_padded, L_inv_padded

	def solve(self, static_verts, handle_verts, handle_verts_pos, mesh_idx=0, n_its=1,
			  track_energy = False, report=False):
		"""apply deforming to mesh

		p = initial mesh deformation
		p0 = working guess

		:param static_verts: list of all vertices which do not move
		:param handle_verts: list of all vertices which are moved as input. Size H
		:param handle_verts_pos: (H x 3) array of target positions of all handle_verts
		:param mesh_idx: index of self for selected mesh.
		:param track_energy: Flag to print energy after every it
		:param report: Flag to use tqdm bar to track iteration progress
"""

		N = self.num_verts_per_mesh()[mesh_idx]
		p = self.verts_padded()[mesh_idx]  # initial mesh

		# Initial guess using Naive Laplacian editing: least square minimisation of |Lp0 - Lp|, subject to known
		# constraints on the values of p, from static and handles
		w = self.w[mesh_idx]
		L = self.L[mesh_idx]

		known_handles = {i:pos for i,pos in zip(handle_verts, handle_verts_pos)}
		known_static = {v:p[v] for v in static_verts}
		known = {**known_handles, **known_static}

		p_prime = least_sq_with_known_values(L, torch.mm(L, p), known=known)
		# TODO: address poor conditioning leading to some large deviations in value

		if n_its == 0:  # if only want initial guess, end here
			self.offset_verts_(p_prime - p)
			return None

		## modify L, L_inv and b_fixed to incorporate boundary conditions
		known_verts = np.array([n for n in range(N) if n in known])
		known_verts_pos = torch.stack([known[v] for v in known_verts]) # Tensor of (n_known x 3) positions
		unknown_verts = [n for n in range(N) if n not in known] # indices of all unknown verts
		n_unknown = len(unknown_verts)
		n_known = len(known)

		b_fixed = torch.zeros((N, 3))  # factor to be subtracted from b, due to constraints
		for k, pos in known.items():
			b_fixed += torch.einsum("i,j->ij", L[:, k], pos)#[unknown]
		L_unknown = L.clone()[np.ix_(unknown_verts, unknown_verts)] # sample sub laplacian matrix for unknowns only
		L_inv_unknown = cholesky_invert(L_unknown)


		## PRECOMPUTE ADDITIONAL PARAMETERS - PD (for rotational equations)
		cell_size_max = max(map(len, self.one_ring_neighbours[mesh_idx].values())) # maximum size of a cell
		PD = torch.zeros((N, 3, cell_size_max)) # This is a padded Tensor, that includes Pi * Di for each cell
		for i in range(N):
			j = self.one_ring_neighbours[mesh_idx][i]
			Pi = (p[i] - p[j]).T
			Di = torch.diag(w[i][j])
			PD[i, :, :len(j)] = torch.mm(Pi, Di)

		# Iterate through method
		if report: progress = tqdm(total=n_its)
		for it in range(n_its):

			# Precompute all P_primes at each iteration
			# full = p_prime.unsqueeze(0).repeat(N, 1, 1) # tensor where each row is [p1, p2, p3, ...]
			# P_prime = (full.permute(1,0,2) - full)

			R = torch.zeros((N, 3, 3))  # Local rotations R
			for i in range(N):
				j = self.one_ring_neighbours[mesh_idx][i]

				P_prime_i_T = (p_prime[i] - p_prime[j])

				Si = torch.mm(PD[i, :, :len(j)], P_prime_i_T)
				Ui, _, Vi = torch.svd(Si)

				R[i] = torch.mm(Vi, Ui.T)


			b = torch.zeros((N, 3)) # RHS of minimum energy equation, with known p's removed
			for i in range(N):
				j = self.one_ring_neighbours[mesh_idx][i]
				deforms = torch.bmm(R[i] + R[j], (p[i] - p[j]).unsqueeze(-1)).squeeze(-1)  # ( N x 3)
				b[i] = 0.5 * torch.mv(deforms.T, w[i,j]) # weighted product of deforms

			b -= b_fixed  # subtract component of LHS not included - constraints

			p_prime_unknown = torch.mm(L_inv_unknown, b[unknown_verts])  # predicted p's for only unknown values

			p_prime = torch.zeros_like(p_prime)  # generate next iteration of fit, from p0_unknown and constraints
			for index, val in known.items():
				p_prime[index] = val

			# Assign initially unknown values to x_out
			p_prime[unknown_verts] = p_prime_unknown

			# track energy
			if track_energy:
				E = 0
				for i in range(N):
					j = self.one_ring_neighbours[mesh_idx][i]
					batch_R = R[i].unsqueeze(0).repeat(len(j), 1, 1) # convert R into batch tensor
					stretch = torch.norm(((p_prime[i] - p_prime[j]).unsqueeze(-1) - torch.bmm(batch_R, (p[i]-p[j]).unsqueeze(-1))), p=2,dim=1).squeeze(-1)**2
					E += torch.dot(w[i,j], stretch)

				print(f"It = {it}, Energy = {E:.2f}")
			# update tqdm
			if report:
				progress.update()

		self.offset_verts_(p_prime - p) # apply transformation in place

	def rotate(self, mesh_idx=0, rot_x=0, rot_y=0, rot_z=0):
		"""Rotate mesh i, in place, in cardinal directions"""

		verts_0 = self.verts_padded()[mesh_idx]

		verts_rotated = verts_0.clone()
		# prerotate in each direction
		for i, theta in enumerate([rot_x, rot_y, rot_z]):
			if theta != 0:
				R = torch.eye(3)
				j, k = (i+1)%3, (i+2)%3 # other two of cyclic triplet
				R[j,j] = R[k,k] = np.cos(theta)
				R[j,k] = - np.sin(theta)
				R[k,j] = np.sin(theta)

				C = self.C[mesh_idx] # centre of mass of mesh

				verts_rotated = np.matmul(R, (verts_rotated-C).T).T - C

		self.offset_verts_((verts_rotated - verts_0))
