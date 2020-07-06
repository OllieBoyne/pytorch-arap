"""Demos the use of ARAP energy as a loss function"""

import numpy as np
from pytorch_arap.arap import ARAP_from_meshes, add_one_ring_neighbours,add_n_ring_neighbours
from pytorch_arap.arap import compute_energy as arap_loss
from pytorch3d.io import load_objs_as_meshes
import os
import torch
from matplotlib import pyplot as plt

from pytorch_arap.arap_utils import save_animation, plot_meshes, profile_backwards
from tqdm import tqdm

if torch.cuda.is_available():
	device = "cuda"
else:
	device = "cpu"

class Model(torch.nn.Module):
	"""Verts to be optimised"""

	def __init__(self, meshes, verts, device="cuda"):

		super().__init__()

		self.device = device
		self.verts_template = verts
		self.meshes = meshes

		self.verts = torch.nn.Parameter(verts.clone())

		self.handle_verts = None
		self.handle_pos = None

	def set_target(self, handle_verts, handle_pos):
		self.handle_verts = handle_verts
		self.handle_pos = handle_pos

	def forward(self):

		loss_target = torch.nn.functional.mse_loss(self.verts[0,self.handle_verts], self.handle_pos)
		loss_arap = arap_loss(self.meshes, self.verts_template, self.verts, device=self.device)

		loss = loss_target + 0.001 * loss_arap

		# print(loss_target, loss_arap, ((self.verts_template-self.verts)**2).mean() )

		return loss

def deform_smal():

	targ = os.path.join("sample_meshes", "smal.obj")
	meshes = load_objs_as_meshes([targ], load_textures=False)
	meshes = ARAP_from_meshes(meshes, device=device) # convert to ARAP obejct
	N = meshes.num_verts_per_mesh()[0]

	meshes.rotate(mesh_idx=0, rot_x=np.pi/2)

	# handle as topmost vert
	handle_verts = [28] # [79]
	handle_verts = add_one_ring_neighbours(meshes, handle_verts)
	handle_pos = meshes.verts_padded()[0][handle_verts]
	handle_pos_shifted = handle_pos.clone()

	# static as base
	static_verts = [1792, 3211, 95, 3667] # centres of paws
	static_verts = add_n_ring_neighbours(meshes, static_verts, n = 6)

	faces = meshes.faces_list()

	prop = True
	trisurfs = plot_meshes(ax, meshes.verts_list(), faces, handle_verts=handle_verts, static_verts=static_verts, prop=prop, change_lims=True,
						   color="gray", zoom=1.5)

	disp_vec = torch.FloatTensor([1, 0, 0])  # displace in x direction

	disp_frac = 0.6 # fraction of full disp_vec to move in animation
	step = disp_frac # moves

	handle_pos_shifted[:] += step * disp_vec

	verts_template = meshes.verts_padded()

	model = Model(meshes, verts_template, device=device)
	model.set_target(handle_verts, handle_pos_shifted)

	# model()

	optimiser = torch.optim.Adam(model.parameters(), lr = 5e-3)

	n_frames = 50
	progress = tqdm(total=n_frames)

	def anim(i):
		[x.remove() for x in trisurfs] # remove previous frame's mesh

		optimiser.zero_grad()
		loss = model()

		loss.backward()
		optimiser.step()

		trisurfs[:] = plot_meshes(ax, model.verts, faces, handle_verts=handle_verts, static_verts=static_verts, prop=prop,
								  color="gray")

		progress.n = i+1
		progress.last_print_n = i+1

		progress.set_description(f"Loss = {loss:.4f}")

	# anim(1)
	# plt.show()

	ax.axis("off")
	save_animation(fig, anim, n_frames, fmt="gif", fps=15, title="smal_arap_lossfit", callback=False)

if __name__ == "__main__":

	fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

	deform_smal()
