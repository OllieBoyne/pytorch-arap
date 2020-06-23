"""Demos the use of ARAP energy as a loss function"""



import numpy as np
from pytorch_arap.arap import ARAP_from_meshes, add_one_ring_neighbours,add_n_ring_neighbours
from pytorch_arap.arap import compute_energy_new as arap_loss
from pytorch3d.io import load_objs_as_meshes
import os
import torch
from matplotlib import pyplot as plt

from pytorch_arap.arap_utils import save_animation, plot_meshes

def deform_smal():

	targ = os.path.join("sample_meshes", "smal.obj")
	meshes = load_objs_as_meshes([targ], load_textures=False)
	meshes = ARAP_from_meshes(meshes) # convert to ARAP obejct
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

	n_frames = 100
	disp_frac = 0.2 # fraction of full disp_vec to move in animation
	step = disp_frac * 4/n_frames # moves


	handle_pos_shifted[:] += step * disp_vec

	## deform, replot
	verts = meshes.solve(static_verts=static_verts, handle_verts=handle_verts, handle_verts_pos=handle_pos_shifted, n_its = 1,
				 track_energy=False) ## run ARAP

	verts_template = meshes.verts_padded()
	verts = verts.unsqueeze(0)

	verts_template.requires_grad = True
	verts.requires_grad = True

	loss = arap_loss(meshes, verts_template, verts)
	print("STARTING BACKWARD")
	loss.backward()
	print("FINISHED BACKWARD")

	trisurfs[:] = plot_meshes(ax, verts, faces, handle_verts=handle_verts, static_verts=static_verts, prop=prop,
							  color="gray")

	ax.axis("off")
	# plt.show()

if __name__ == "__main__":

	fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

	deform_smal()
