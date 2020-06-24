import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss.mesh_laplacian_smoothing import laplacian_cot
from collections import defaultdict

import sys
sys.path.append("../")
from .arap_utils import least_sq_with_known_values, Timer
from tqdm import tqdm


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

		:param optimise: flag if mesh will be used for optimisation
		"""

		super().__init__(verts, faces, textures)

		self.one_ring_neighbours = self.get_one_ring_neighbours()

		self.C = self.verts_padded().mean(dim=1) # centre of mass for each mesh
		self.timer = Timer()

		self.precomputed_params = {} # dictionary to store precomputed parameters


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

	def get_cot_padded(self) -> torch.Tensor:
		"""Retrieves cotangent weights 0.5 * cot a_ij + cot b_ij for each mesh. Returns as a padded tensor.

		:return cot_weights: Tensor of cotangent weights, shape (N_mesh, max(n_verts_per_mesh), max(n_verts_per_mesh))
		:type cot_weights: Tensor"""

		w_packed, _ = laplacian_cot(self)  # per-edge weightings used
		w_packed = w_packed.to_dense()

		# w_packed[w_packed<0] = 0

		n = self.num_verts_per_mesh()
		max_n = max(n.tolist())
		w_padded = torch.zeros((len(self), max_n, max_n))

		# extract w for each mesh
		cum_n = np.concatenate([[0], np.cumsum(n.tolist())])  # list of n_verts before and after every mesh is included

		for i, (start, end) in enumerate(zip(cum_n[:-1], cum_n[1:])):
			w_padded[i] = w_packed[start:end, start:end]

		return w_padded

	def precompute_laplacian(self):
		"""Precompute edge weights and Laplacian-Beltrami operator"""

		N = self.num_verts_per_mesh()
		p = self.verts_padded()  # undeformed mesh

		w_padded = self.get_cot_padded()

		L_padded = torch.zeros_like(w_padded)
		L_inv_padded = torch.zeros_like(w_padded)
		for n in range(len(self)):
			w = w_padded[n]
			# diagonal L_ii = sum(w_i). non-diagonal L_ij = -w_ij

			L_padded[n] = torch.diag(torch.sum(w, dim=1)) - w
			# ii = np.arange(N[n])
			# L_padded[ii] = - w_

			L_inv_padded[n] = torch.cholesky(L_padded[n])

		self.precomputed_params["L_padded"] = L_padded
		self.precomputed_params["w_padded"] = w_padded
		self.precomputed_params["L_inv_padded"] = L_inv_padded


	def precompute_reduced_laplacian(self, static_verts, handle_verts):
		"""Precompute the Laplacian-Beltrami operator for the reduced set of vertices, negating static and handle verts"""

		L = self.precomputed_params["L_padded"][0]
		N = self.num_verts_per_mesh()[0]
		unknown_verts = [n for n in range(N) if n not in static_verts + handle_verts] # all unknown verts
		L_reduced = L.clone()[np.ix_(unknown_verts, unknown_verts)] # sample sub laplacian matrix for unknowns only
		L_reduced_inv = cholesky_invert(L_reduced)

		self.precomputed_params["L_reduced_inv"] = L_reduced_inv

	def precompute_PD(self):
		"""Precompute the product of the undeformed edges and laplacian weights - PD. Used to calculate
		the rigid body relations R"""

		N = self.num_verts_per_mesh()[0]
		p = self.verts_padded()[0]  # undeformed verts
		w = self.precomputed_params["w_padded"][0]

		cell_size_max = max(map(len, self.one_ring_neighbours[0].values())) # maximum size of a cell
		PD = torch.zeros((N, 3, cell_size_max)) # This is a padded Tensor, that includes Pi * Di for each cell
		for i in range(N):
			j = self.one_ring_neighbours[0][i]
			Pi = (p[i] - p[j]).T
			Di = torch.diag(w[i][j])
			PD[i, :, :len(j)] = torch.mm(Pi, Di)

		self.precomputed_params["PD"] = PD

	def solve(self, static_verts, handle_verts, handle_verts_pos, mesh_idx=0, n_its=1,
			  track_energy = False, report=False):
		"""
		Solve iterations of the As-Rigid-As-Possible method.

		:param static_verts: list of all vertices which do not move
		:param handle_verts: list of all vertices which are moved as input. Size H
		:param handle_verts_pos: (H x 3) array of target positions of all handle_verts
		:param mesh_idx: index of self for selected mesh.
		:param track_energy: Flag to print energy after every it
		:param report: Flag to use tqdm bar to track iteration progress

		p = initial mesh deformation
		p0 = working guess
"""

		N = self.num_verts_per_mesh()[mesh_idx]
		p = self.verts_padded()[mesh_idx]  # initial mesh

		if "w_padded" not in self.precomputed_params:
			self.precompute_laplacian()

		w = self.precomputed_params["w_padded"][mesh_idx]
		L = self.precomputed_params["L_padded"][mesh_idx]

		known_handles = {i:pos for i,pos in zip(handle_verts, handle_verts_pos)}
		known_static = {v:p[v] for v in static_verts}
		known = {**known_handles, **known_static}

		# Initial guess using Naive Laplacian editing: least square minimisation of |Lp0 - Lp|, subject to known
		# constraints on the values of p, from static and handles
		p_prime = least_sq_with_known_values(L, torch.mm(L, p), known=known)
		# TODO: address poor conditioning leading to some large deviations in value

		if n_its == 0:  # if only want initial guess, end here
			return p_prime

		## modify L, L_inv and b_fixed to incorporate boundary conditions
		known_verts = np.array([n for n in range(N) if n in known])
		unknown_verts = [n for n in range(N) if n not in known] # indices of all unknown verts
		n_unknown = len(unknown_verts)
		n_known = len(known)

		b_fixed = torch.zeros((N, 3))  # factor to be subtracted from b, due to constraints
		for k, pos in known.items():
			b_fixed += torch.einsum("i,j->ij", L[:, k], pos) # [unknown]

		#  Precompute L_reduced_inv if not already done
		if "L_reduced_inv" not in self.precomputed_params:
			self.precompute_reduced_laplacian(static_verts, handle_verts)

		#  Precompute PD if not already done
		if "PD" not in self.precomputed_params:
			self.precompute_PD()

		L_reduced_inv = self.precomputed_params["L_reduced_inv"]
		PD = self.precomputed_params["PD"]

		# Iterate through method
		if report: progress = tqdm(total=n_its)

		for it in range(n_its):
			R = torch.zeros((N, 3, 3))  # Local rotations R
			for i in range(N):
				j = self.one_ring_neighbours[mesh_idx][i]

				P_prime_i_T = (p_prime[i] - p_prime[j])

				Si = torch.mm(PD[i, :, :len(j)], P_prime_i_T)
				Ui, sig, Vi = torch.svd(Si)
				Ri = torch.mm(Vi, Ui.T)

				if torch.det(Ri) <= 0:
					# Flip column of Ui corresponding to smallest singular value (ensures det(Ri)>0)
					col_flip = (sig == sig.min()).nonzero()[0, 0]
					Ui_mod = Ui.clone()
					Ui_mod[:, col_flip] *= -1
					R[i] = torch.mm(Vi, Ui.T)

				else:
					R[i] = Ri

			b = torch.zeros((N, 3)) # RHS of minimum energy equation, with known p's removed
			for i in unknown_verts:
				j = self.one_ring_neighbours[mesh_idx][i]
				deforms = torch.bmm(R[i] + R[j], (p[i] - p[j]).unsqueeze(-1)).squeeze(-1)  # ( len(j) x 3)
				b[i] = 0.5 * torch.mv(deforms.T, w[i,j])  # weighted product of deforms (3)

			# self.timer.add("b")
			b -= b_fixed  # subtract component of LHS not included - constraints
			p_prime_unknown = torch.mm(L_reduced_inv, b[unknown_verts])  # predicted p's for only unknown values

			p_prime = torch.zeros_like(p_prime)  # generate next iteration of fit, from p0_unknown and constraints
			for index, val in known.items():
				p_prime[index] = val

			# Assign initially unknown values to x_out
			p_prime[unknown_verts] = p_prime_unknown

			# track energy
			if track_energy:
				energy = compute_energy(self,[p], [p_prime])
				print(f"It = {it}, Energy = {energy:.2f}")
			# update tqdm
			if report:
				progress.update()

		return p_prime # return new vertices


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


def get_cot_weights(meshes) -> torch.Tensor:
	"""Given a meshes object, return packed tensor of cotangent weights, where
	w_ij = 0.5 * (alpha_ij + beta_ij)"""

	w = laplacian_cot(meshes)[0]
	w = w.to_dense() / 2

	## split to padded
	n = meshes.num_verts_per_mesh()
	w_padded = torch.zeros((len(meshes), n.max(), n.max()))
	cum_n = np.concatenate([[0], np.cumsum(n.tolist())])  # list of n_verts before and after every mesh is included

	for i, (start, end) in enumerate(zip(cum_n[:-1], cum_n[1:])):
		w_padded[i] = w[start:end, start:end]

	return w_padded

def get_cot_weights_full(meshes, verts=None) -> torch.Tensor:
	"""Given a meshes object, return a padded tensor w, of shape (N_meshes, max_verts, max_verts),
	where, for a given mesh, the tensor w:
	w_ij = 0.5 * ( cot(alpha_ij) + cot(beta_ij) )
	where alpha and beta are the angles on the faces adjacent to the edge, which are opposite the edge itself.

	Derived from pyTorch3D's laplacian_cot method"""

	n_meshes = len(meshes)
	all_n_verts = meshes.num_verts_per_mesh()
	all_n_faces = meshes.num_faces_per_mesh()
	max_verts = max(all_n_verts.tolist())

	all_faces = meshes.faces_padded()
	if verts is None:
		all_verts = meshes.verts_padded()
	else:
		all_verts = verts

	w = torch.zeros((n_meshes, max_verts, max_verts))

	for n in range(n_meshes):
		V, F = all_n_verts[n], all_n_faces[n]
		verts = all_verts[n]
		faces = all_faces[n]

		face_verts = verts[faces]  # (F x 3) of verts per face
		v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

		# Side lengths of each triangle, of shape (sum(F_n),)
		# A is the side opposite v1, B is opposite v2, and C is opposite v3
		A = (v1 - v2).norm(dim=1)
		B = (v0 - v2).norm(dim=1)
		C = (v0 - v1).norm(dim=1)

		# Area of each triangle (with Heron's formula); shape is (F)
		s = 0.5 * (A + B + C)
		# note that the area can be negative (close to 0) causing nans after sqrt()
		# we clip it to a small positive value
		area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

		# Compute cotangents of angles, of shape (F, 3)
		A2, B2, C2 = A * A, B * B, C * C
		cota = (B2 + C2 - A2) / area
		cotb = (A2 + C2 - B2) / area
		cotc = (A2 + B2 - C2) / area
		cot = torch.stack([cota, cotb, cotc], dim=1)
		cot /= 4.0

		i = faces[:, [0, 1, 2]].view(-1)  # flattened tensor of by face, v0, v1, v2
		j = faces[:, [1, 2, 0]].view(-1)  # flattened tensor of by face, v1, v2, v0

		# flatten cot, such that the following line sets
		# w_ij = 0.5 * cot a_ij
		w[n][i, j] = 0.5 * cot.view(-1)
		# to include b_ij, simply add the transpose to itself
		w[n] += w[n].T

	return w


def produce_edge_matrix(verts: torch.Tensor) -> torch.Tensor:
	"""Given a tensor of verts postion, p (V x 3), produce a tensor E, where
	E_ij = p_i - p_j"""

	V, _ = verts.shape
	pi = verts.unsqueeze(0).repeat(V, 1, 1) # tensor where each row is the same
	pj = pi.permute(1, 0, 2) # tensor where each column is the same

	E = pi - pj
	return E

def compute_energy(meshes: ARAPMeshes, verts: torch.Tensor, verts_deformed: torch.Tensor, mesh_idx = 0):
	"""Compute the energy of a deformation for a deformation, according to

	sum_i w_i * sum_j w_ij || (p'_i - p'_j) - R_i(p_i - p_j) ||^2

	Where i is the vertex index,
	j is the indices of all one-ring-neighbours
	p gives the undeformed vertex locations
	p' gives the deformed vertex rotations
	and R gives the rotation matrix between p and p' that captures as much of the deformation as possible
	(maximising the amount of deformation that is rigid)

	w_i gives the per-cell weight, selected as 1
	w_ij gives the per-edge weight, selected as 0.5 * (cot (alpha_ij) + cot(beta_ij)), where alpha and beta
	give the angles opposite of the mesh edge

	:param meshes: ARAP meshes object
	:param verts_deformed:
	:param verts:

	:return energy: Tensor of strain energy of deformation

	"""

	N = meshes.num_verts_per_mesh()[mesh_idx]
	w = get_cot_weights_full(meshes, verts)[mesh_idx]

	p = verts[mesh_idx]  # initial mesh
	p_prime = verts_deformed[mesh_idx] # displaced verts

	P = produce_edge_matrix(p)
	P_prime = produce_edge_matrix(p_prime)

	R = torch.zeros((N, 3, 3))  # Local rotations R
	for i in range(N):
		J = meshes.one_ring_neighbours[0][i]

		P_i = (p[i] - p[J]).T
		D_i = torch.diag(w[i,J])
		P_prime_i = (p_prime[i] - p_prime[J]).T

		## in the case of no deflection, set R = I. This is to avoid numerical errors
		if torch.equal(P_i, P_prime_i):
			R[i] = torch.eye(3)
			continue

		# Get rigid rotation that maximises Tr(R * (PDP'))
		Si = torch.mm(P_i, torch.mm(D_i, P_prime_i.T))
		Ui, sig, Vi = torch.svd(Si)

		Ri = torch.mm(Vi, Ui.T)

		if torch.det(Ri) <= 0:
			# Flip column of Ui corresponding to smallest singular value (ensures det(Ri)>0)
			col_flip = (sig == sig.min()).nonzero()[0, 0]
			Ui_mod = Ui.clone()
			Ui_mod[:, col_flip] *= -1
			R[i] = torch.mm(Vi, Ui_mod.T)

		else:
			R[i] = Ri

	def sparse_dense_mul(s, d):
		i = s._indices()
		v = s._values()
		dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
		return torch.sparse.FloatTensor(i, v * dv, s.size())

	stretch = P_prime - torch.bmm(R, P.permute(0, 2, 1)).permute(0, 2, 1)
	stretch_norm = torch.norm(stretch, dim=2)**2  # norm over (x,y,z) space
	energy = (w * stretch_norm).sum() # element wise multiplication

	return energy