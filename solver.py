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

def ARAP_from_meshes(meshes):
	"""Produce ARAPMeshes object from Meshes object
	:type meshes: Meshes
	"""
	verts = meshes.verts_list()
	faces = meshes.faces_list()
	textures = meshes.textures

	return ARAPMeshes(verts=verts, faces=faces, textures=textures)

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

		self.L, self.w = self.precompute()  # Precomputed edge weights and Laplace-Beltrami operato

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

	def mesh_to_numpy(self, i=0):
		"""
		Given a mesh index i, return the verts, faces for that mesh, as numpy arrays
		:return: verts (N x 3) ndarray, faces (Mx3) ndarray
		"""
		verts = self.verts_list()[i].detach().numpy()
		faces = self.faces_list()[i].detach().numpy()

		return verts, faces

	def precompute(self):
		"""Precompute edge weights and Laplacian-Beltrami operator"""

		ws, Ls = [], []

		n = self.num_verts_per_mesh()
		p = self.verts_padded()  # undeformed mesh

		# pre-factored edge weightings, and Laplace-Beltrami operator
		w_packed, _ = laplacian_cot(self)  # per-edge weightings used
		w_packed = w_packed.to_dense()
		# diagonal L_ii = sum(w_i). non-diagonal L_ij = -w_ij
		L_packed = torch.diag(torch.sum(w_packed, dim=1)) - w_packed

		## w is packed currently. Convert to padded
		max_n = max(n.tolist())
		w_padded = torch.zeros((len(self), max_n, max_n))
		L_padded = torch.zeros((len(self), max_n, max_n))

		# extract w for each mesh
		cum_n = np.concatenate([[0], np.cumsum(n.tolist())])  # list of n_verts before and after every mesh is included

		for i, (start, end) in enumerate(zip(cum_n[:-1], cum_n[1:])):
			w_padded[i] = w_packed[start:end, start:end]
			L_padded[i] = L_packed[start:end, start:end]

		return L_padded, w_padded

	def arap_iteration(self, p1, static_verts, handle_verts, handle_verts_pos, mesh_idx=0):
		"""Solves a single iteration of the ARAP problem.

		:param p1: (Nx3) array of initial guess of all vertex positions
		:param static_verts: list of all vertices which do not move
		:param handle_verts: list of all vertices which are moved as input. Size H
		:param handle_verts_pos: (H x 3) array of target positions of all handle_verts
		:param mesh_idx: index of self for selected mesh.

		:returns: p_new - (Nx3) iterative guess of best vertex placements

		Finds a set of vertex positions p that better matches the minimum energy condition,

		<--- eq8 --->

		"""
		mesh = self[mesh_idx]

		n = self.num_verts_per_mesh()[mesh_idx]
		p = self.verts_padded()[mesh_idx] # undeformed mesh

		w = self.w[mesh_idx]
		L = self.L[mesh_idx]

		# compute local rotations R
		R = torch.zeros((n, 3, 3))
		for i in range(n):
			j = self.one_ring_neighbours[mesh_idx][i]
			Pi = (p[i] - p[j]).T
			Di = torch.diag(w[i][j])
			P1i = (p1[i] - p1[j]).T

			Si = torch.mm(Pi, torch.mm(Di, P1i.T))
			Vi, _, Ui = torch.svd(Si)

			R[i] = torch.mm(Vi, Ui.T)

		# RHS for minimum energy equation
		# N = number of neighbours
		b = torch.zeros((n, 3))
		for i in range(n):
			j = self.one_ring_neighbours[mesh_idx][i]

			deforms = torch.bmm(R[i]+R[j], (p[i]-p[j]).unsqueeze(-1) ).squeeze(-1) # ( N x 3)
			b[i] = 0.5 * torch.einsum("j, ji -> i", w[i][j], deforms)

		## apply constraints
		for v in static_verts:
			pos = p[v]
			L[v] *= 0
			L[v, v] = 1
			b[v] = pos

		for i, v in enumerate(handle_verts):
			pos = handle_verts_pos[i]
			L[v] *= 0
			L[v, v] = 1
			b[v] = pos


		# L_L = torch.cholesky(L) # lower triangular matrix for L
		# L_L_inv = torch.inverse(L_L)
		# L_inv = torch.mm(L_L_inv.T, L_L_inv)
		t0 = perf_counter()
		L_inv = torch.inverse(L)

		p_new = torch.mm(L_inv, b)
		return p_new

	def solve(self, static_verts, handle_verts, handle_verts_pos, mesh_idx=0, n_its=1):
		"""apply deforming to mesh"""

		p = self.verts_padded()[mesh_idx]  # initial mesh
		handle_verts_pos = torch.from_numpy(handle_verts_pos).float() # convert handle verts pos to numpy array

		# make first 'guess' - initial mesh with handles deformed
		p0 = p.clone()
		# p0[handle_verts] = handle_verts_pos

		for i in range(n_its):
			p0 = self.arap_iteration(p0, static_verts, handle_verts, handle_verts_pos, mesh_idx=mesh_idx)

		## apply transformation
		self.offset_verts_(p0 - p)



