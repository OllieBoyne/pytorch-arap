"""Operations used for calculating ARAP

jfmt is a standard Tensor index notation, of i, j, where i = vert idx, and j = neighbouring vert idx

nfmt is a Tensor index method used to drastically reduce the size of the tensors used. Rather than use sparse matrices, the index notation for
what would usually be i, j in jfmt, is instead transformed to i, n
where n is the index of j within the list of neighbours J for every i.
For each mesh, n is fixed in the range 0 < n < N, where N is the maximum size of all J for that mesh
"""

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss.mesh_laplacian_smoothing import laplacian_cot
from collections import defaultdict

import sys
sys.path.append("../")
from .arap_utils import least_sq_with_known_values, Timer
from tqdm import tqdm

### Attempt to import svd batch method. If not provided, use default method
### Sourced from https://github.com/KinglittleQ/torch-batch-svd/blob/master/torch_batch_svd/include/utils.h
try:
	from torch_batch_svd import svd as batch_svd
except ImportError:
	print("torch_batch_svd not installed. Using torch.svd instead")
	batch_svd = torch.svd

def ARAP_from_meshes(meshes, device="cuda"):
	"""Produce ARAPMeshes object from Meshes object
	:type meshes: Meshes
	"""
	verts = meshes.verts_list()
	faces = meshes.faces_list()
	textures = meshes.textures

	return ARAPMeshes(verts=verts, faces=faces, textures=textures, device=device)

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

def cholesky_invert(A):

	L = torch.cholesky(A)
	L_inv = torch.inverse(L)
	A_inv = torch.mm(L_inv.T, L_inv)

	return A_inv

class ARAPMeshes(Meshes):
	"""
	Subclass of PyTorch3D Meshes.
	Upon deformation, provided with 'static' vertices, and 'handle' vertices.
	Allows for movement of handle vertices from original mesh position, and identifies new positions of all other
	non-static verts using the As Rigid As Possible algorithm (As-Rigid-As-Possible Surface Modeling, O. Sorkine & M. Alexa)"""

	def __init__(self, verts=None, faces=None, textures=None, device="cuda"):
		"""
		lists of verts, faces and textures for methods. For details, see Meshes documentation

		:param optimise: flag if mesh will be used for optimisation
		"""

		super().__init__(verts, faces, textures)

		self.one_ring_neighbours = self.get_one_ring_neighbours()

		self.C = self.verts_padded().mean(dim=1) # centre of mass for each mesh
		self.timer = Timer()

		self.precomputed_params = {} # dictionary to store precomputed parameters

		# Precompute cotangent weights in nfmt. nfmt is defined at the top of this script
		w_full = get_cot_weights_full(self, device=device)
		w_nfmts = produce_cot_weights_nfmt(self, self.one_ring_neighbours, device=device)
		self.w_nfmts = w_nfmts

	def get_one_ring_neighbours(self):
		"""Return a dict, where key i gives a list of all neighbouring vertices (connected by exactly one edge)"""

		all_faces = self.faces_padded().cpu().detach().numpy()
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
			L_padded[n] = torch.diag(torch.sum(w, dim=0)) - w
			L_inv_padded[n] = torch.inverse(torch.cholesky(L_padded[n]))

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

		V = self.num_verts_per_mesh()[mesh_idx]
		p = self.verts_padded()[mesh_idx]  # initial mesh

		if "w_padded" not in self.precomputed_params:
			self.precompute_laplacian()

		L = self.precomputed_params["L_padded"][mesh_idx]

		known_handles = {i:pos for i,pos in zip(handle_verts, handle_verts_pos)}
		known_static = {v:p[v] for v in static_verts}
		known = {**known_handles, **known_static}

		# Initial guess using Naive Laplacian editing: least square minimisation of |Lp0 - Lp|, subject to known
		# constraints on the values of p, from static and handles
		p_prime = least_sq_with_known_values(L, torch.mm(L, p), known=known)

		if n_its == 0:  # if only want initial guess, end here
			return p_prime

		## modify L, L_inv and b_fixed to incorporate boundary conditions
		unknown_verts = [n for n in range(V) if n not in known] # indices of all unknown verts

		b_fixed = torch.zeros((V, 3))  # factor to be subtracted from b, due to constraints
		for k, pos in known.items():
			b_fixed += torch.einsum("i,j->ij", L[:, k], pos) # [unknown]

		#  Precompute L_reduced_inv if not already done
		if "L_reduced_inv" not in self.precomputed_params:
			self.precompute_reduced_laplacian(static_verts, handle_verts)

		L_reduced_inv = self.precomputed_params["L_reduced_inv"]

		orn = self.one_ring_neighbours[mesh_idx]
		max_neighbours = max(map(len, orn.values()))  # largest number of neighbours

		ii, jj, nn = produce_idxs(V, orn, self.device)  # flattened tensors for indices
		w = self.w_nfmts[mesh_idx]  # cotangent weight matrix, in nfmt index format

		edge_shape = (V, max_neighbours, 3)
		P = produce_edge_matrix_nfmt(p, edge_shape, ii, jj, nn, device=self.device)

		# Iterate through method
		if report: progress = tqdm(total=n_its)

		for it in range(n_its):

			P_prime = produce_edge_matrix_nfmt(p_prime, edge_shape, ii, jj, nn, device=self.device)

			### Calculate covariance matrix in bulk
			D = torch.diag_embed(w, dim1=1, dim2=2)
			S = torch.bmm(P.permute(0, 2, 1), torch.bmm(D, P_prime))

			## in the case of no deflection, set S = 0, such that R = I. This is to avoid numerical errors
			unchanged_verts = torch.unique(torch.where((P == P_prime).all(dim=1))[0])  # any verts which are undeformed
			S[unchanged_verts] = 0

			U, sig, W = batch_svd(S)
			R = torch.bmm(W, U.permute(0, 2, 1))  # compute rotations

			# Need to flip the column of U corresponding to smallest singular value
			# for any det(Ri) <= 0
			entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
			if len(entries_to_flip) > 0:
				Umod = U.clone()
				cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
				Umod[entries_to_flip, :, cols_to_flip] *= -1  # flip cols
				R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))

			### RHS of minimum energy equation
			Rsum_shape = (V, max_neighbours, 3, 3)
			Rsum = torch.zeros(Rsum_shape).to(self.device) # Ri + Rj, as in eq (8)
			Rsum[ii, nn] = R[ii] + R[jj]

			### Rsum has shape (V, max_neighbours, 3, 3). P has shape (V, max_neighbours, 3)
			### To batch multiply, collapse first 2 dims into a single batch dim
			Rsum_batch, P_batch = Rsum.view(-1, 3, 3), P.view(-1,3).unsqueeze(-1)

			# RHS of minimum energy equation
			b = 0.5 * (w[..., None] * torch.bmm(Rsum_batch, P_batch).squeeze(-1).reshape(V, max_neighbours, 3)).sum(dim=1)

			b -= b_fixed  # subtract component of LHS not included - constraints

			p_prime_unknown = torch.mm(L_reduced_inv, b[unknown_verts])  # predicted p's for only unknown values

			p_prime = torch.zeros_like(p_prime)  # generate next iteration of fit, from p0_unknown and constraints
			for index, val in known.items():
				p_prime[index] = val

			# Assign initially unknown values to x_out
			p_prime[unknown_verts] = p_prime_unknown

			# track energy
			if track_energy:
				energy = compute_energy(self,[p], [p_prime], device=self.device)
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


def get_cot_weights_full(meshes, verts=None, device="cuda", sparse=False) -> torch.Tensor:
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

	if sparse:
		W = []
	
	else:
		W = torch.zeros((n_meshes, max_verts, max_verts)).to(device)

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

		if sparse:
			ii = faces[:, [1, 2, 0]]
			jj = faces[:, [2, 0, 1]]
			idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
			w = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))
			w += w.t()
			W.append(w)

		else:
			i = faces[:, [0, 1, 2]].view(-1)  # flattened tensor of by face, v0, v1, v2
			j = faces[:, [1, 2, 0]].view(-1)  # flattened tensor of by face, v1, v2, v0

			# flatten cot, such that the following line sets
			# w_ij = 0.5 * cot a_ij
			W[n][i, j] = 0.5 * cot.view(-1)
			# to include b_ij, simply add the transpose to itself
			W[n] += W[n].T

	return W


def produce_cot_weights_nfmt(meshes, orn_list, verts=None, device="cuda", mesh_idx = 0):
	"""Convert a tensor w_ij of cotangent weights, to a format w_in, where
	w_in = 0.5 * (a_ij + b_ij), where j = J[n].
	
	:returns w_nfmt: list size n_meshes, each with tensor of shape (verts, max_neighbours_in_mesh)
	"""

	w_nfmt = []

	w_full = get_cot_weights_full(meshes, device=device)

	for mesh_idx in range(len(meshes)):
		orn = orn_list[mesh_idx]
		max_neighbours = max(map(len, orn.values())) # largest number of neighbours
		V = meshes.num_verts_per_mesh()[mesh_idx]
		W = w_full[mesh_idx]

		Wn = torch.zeros((V, max_neighbours)).to(device)
		ii, jj, nn = produce_idxs(V, orn, device)

		Wn[ii, nn] = W[ii, jj]
		w_nfmt.append(Wn)

	return w_nfmt

def produce_edge_matrix_nfmt(verts: torch.Tensor, edge_shape, ii, jj, nn, device="cuda") -> torch.Tensor:
	"""Given a tensor of verts postion, p (V x 3), produce a tensor E, where, for neighbour list J,
	E_in = p_i - p_(J[n])"""

	E = torch.zeros(edge_shape).to(device)
	E[ii, nn] = verts[ii] - verts[jj]

	return E

def produce_idxs(V, orn, device = "cuda"):
	"""For V verts and dict of one ring neighbours, return indices of every i, j and n
	where i = vert idx, j = neighbouring vert idx, n = number neighbour"""

	ii = []
	jj = []
	nn = []
	for i in range(V):
		J = orn[i]
		for n, j in enumerate(J):
			ii.append(i)
			jj.append(j)
			nn.append(n)

	ii = torch.LongTensor(ii).to(device)
	jj = torch.LongTensor(jj).to(device)
	nn = torch.LongTensor(nn).to(device)
	
	return ii, jj, nn

def produce_edge_matrix(verts: torch.Tensor, orn: dict, device="cuda") -> torch.Tensor:
	"""Given a tensor of verts postion, p (V x 3), produce a sparse tensor E, where
	E_ij = p_i - p_j if j and i are neighbours
	E_ij = 0 otherwise"""
	V, _ = verts.shape

	ii, jj, _ = produce_idxs(V, orn, device)
	idx = torch.stack([ii, jj], dim=0).to(device)

	values = verts[ii] - verts[jj]
	E = torch.sparse.FloatTensor(idx, values, (V, V, 3))#.to(device)

	return E

def compute_energy(meshes: ARAPMeshes, verts: torch.Tensor, verts_deformed: torch.Tensor, mesh_idx = 0, device="cuda"):
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

	V = meshes.num_verts_per_mesh()[mesh_idx]

	orn = meshes.one_ring_neighbours[mesh_idx]
	max_neighbours = max(map(len, orn.values())) # largest number of neighbours

	ii, jj, nn = produce_idxs(V, orn, device) # flattened tensors for indices
	
	w = meshes.w_nfmts[mesh_idx]	# cotangent weight matrix, in nfmt index format

	p = verts[mesh_idx]  # initial mesh
	p_prime = verts_deformed[mesh_idx] # displaced verts

	edge_shape = (V, max_neighbours, 3)
	P = produce_edge_matrix_nfmt(p, edge_shape, ii, jj, nn, device=device)
	P_prime = produce_edge_matrix_nfmt(p_prime, edge_shape, ii, jj, nn, device=device)


	### Calculate covariance matrix in bulk
	D = torch.diag_embed(w, dim1=1, dim2=2)
	S = torch.bmm(P.permute(0,2,1), torch.bmm(D, P_prime))

	## in the case of no deflection, set S = 0, such that R = I. This is to avoid numerical errors
	unchanged_verts = torch.unique(torch.where((P == P_prime).all(dim=1))[0]) # any verts which are undeformed
	S[unchanged_verts] = 0

	U, sig, W = batch_svd(S)
	R = torch.bmm(W, U.permute(0,2,1))	# compute rotations

	# Need to flip the column of U corresponding to smallest singular value
	# for any det(Ri) <= 0
	entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten() # idxs where det(R) <= 0
	if len(entries_to_flip) > 0:
		Umod = U.clone()
		cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1) # Get minimum singular value for each entry
		Umod[entries_to_flip, :, cols_to_flip] *= -1 # flip cols
		R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0,2,1))

	# Compute energy
	rot_rigid = torch.bmm(R, P.permute(0, 2, 1)).permute(0, 2, 1)
	stretch_vec = P_prime - rot_rigid # stretch vector
	stretch_norm = (torch.norm(stretch_vec, dim=2)**2)  # norm over (x,y,z) space
	energy = (w * stretch_norm).sum()

	return energy