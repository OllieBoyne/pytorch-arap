from matplotlib import pyplot as plt

import numpy as np

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss.mesh_laplacian_smoothing import mesh_laplacian_smoothing
from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency, chamfer_distance
from pytorch3d.structures import Meshes

from arap import ARAPMeshes, ARAP_from_meshes, add_one_ring_neighbours,add_n_ring_neighbours, compute_energy
from arap_optimise import ARAPOptimise
from pytorch3d.io import load_obj, load_objs_as_meshes
import os
import torch

from utils import save_animation, plot_meshes
from tqdm import tqdm

nn = torch.nn

def deform_cactus():

	targ = os.path.join("sample_meshes", "cactus.obj")
	meshes = load_objs_as_meshes([targ], load_textures=False)
	meshes = ARAP_from_meshes(meshes) # convert to ARAP obejct
	N = meshes.num_verts_per_mesh()[0]

	# handle as topmost vert
	handle_verts = [504]
	handle_verts = add_one_ring_neighbours(meshes, handle_verts)
	handle_pos = meshes.verts_padded()[0][handle_verts]
	handle_pos_shifted = handle_pos.clone()

	# static as base
	static_verts = [619] # centres of base
	static_verts = add_n_ring_neighbours(meshes, static_verts, n = 2)

	faces = meshes.faces_list()

	prop = True
	# trisurfs = plot_meshes(ax, meshes.verts_list(), faces, handle_verts=handle_verts, static_verts=static_verts, prop=prop, change_lims=True,
	# 					   color="gray", zoom=1.5)

	disp_vec = torch.FloatTensor([1, 0, 0])  # displace in x direction

	disp_frac = 0.5 # fraction of full disp_vec to move in animation
	step = disp_frac # moves

	model = ARAPOptimise(meshes)
	optimiser = torch.optim.Adam(model.parameters(), lr = 1e-3)

	handle_pos_shifted[:] += step * disp_vec

	nits = 10

	with tqdm(np.arange(nits)) as tqdm_iterator:
		for j in tqdm_iterator:
			optimiser.zero_grad()
			loss_arap = model()
			# print(model.verts.shape)

			# rms difference between current handle verts, and target handle verts
			loss_target = torch.norm(model.verts[0, handle_verts] - handle_pos_shifted, dim=1).mean()
			loss_target += torch.norm(model.verts[0, static_verts] - model.undeformed_verts[0, static_verts], dim=1).mean()

			loss = 10 * loss_target + loss_arap

			loss.backward()
			tqdm_iterator.set_description(f"LOSS = {loss:.2f}")
			# print(loss)
			optimiser.step()

	verts = [model.verts[0]]

	plot_meshes(ax, verts, faces, handle_verts=handle_verts, static_verts=static_verts, prop=prop,
							  color="gray", change_lims=True)

	plt.show()

	# [anim(i) for i in range(2)]
	# plt.show()

	# ax.axis("off")
	# save_animation(fig, anim, n_frames=n_frames, title="cactus_optim", fps = 30)

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
	# static_verts += [262] # bottom of mouth
	static_verts = add_n_ring_neighbours(meshes, static_verts, n = 6)

	faces = meshes.faces_list()

	prop = True
	trisurfs = plot_meshes(ax, meshes.verts_list(), faces, handle_verts=handle_verts, static_verts=static_verts, prop=prop, change_lims=True,
						   color="gray", zoom=1.5)

	disp_vec = torch.FloatTensor([1, 0, 0])  # displace in x direction

	n_frames = 10
	disp_frac = 0.2 # fraction of full disp_vec to move in animation
	step = disp_frac * 4/n_frames # moves

	model = ARAPOptimise(meshes)
	optimiser = torch.optim.Adam(model.parameters(), lr = 1e-3)

	def anim(i):
		[x.remove() for x in trisurfs] # remove previous frame's mesh

		# step desired tail position
		handle_pos_shifted[:, 2] += 0.5

		subnits = 2
		for j in range(subnits):
			optimiser.zero_grad()
			loss_energy = model()
			# rms difference between current handle verts, and target handle verts
			loss_target = nn.functional.mse_loss(model.verts[0, handle_verts], handle_pos_shifted)

			loss = loss_target

			loss.backward()
			# print(loss)
			optimiser.step()

		verts = [model.verts[0]]

		trisurfs[:] = plot_meshes(ax, verts, faces, handle_verts=handle_verts, static_verts=static_verts, prop=prop,
								  color="gray")



	# [anim(i) for i in range(2)]
	# plt.show()

	ax.axis("off")
	save_animation(fig, anim, n_frames=n_frames, title="smal_optim", fps = 30)

if __name__ == "__main__":

	fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

	# deform_smal()
	deform_cactus()