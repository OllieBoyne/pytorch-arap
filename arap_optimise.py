"""Class for running optimisation on As Rigid As Possible meshes"""

import torch
nn = torch.nn
from arap import ARAPMeshes, ARAP_from_meshes, compute_energy
from pytorch3d.structures.meshes import Meshes

class ARAPOptimise(nn.Module):

	def __init__(self, meshes):
		"""
		Produce As Rigid As Possible Optimiser.

		:param meshes: PyTorch3D meshes object, or ARAPMeshes object
		"""

		super().__init__()

		if isinstance(meshes, Meshes):
			meshes = ARAP_from_meshes(meshes)

		assert isinstance(meshes, ARAPMeshes), "Meshes must be either a pytorch3d.structures.meshes.Meshes or arap.ARAPMeshes"

		self.meshes = meshes

		self.undeformed_verts = self.meshes.verts_padded()
		self.verts = nn.Parameter(self.meshes.verts_padded())


	def forward(self):

		energy = compute_energy(self.meshes, self.undeformed_verts, self.verts)
		return energy