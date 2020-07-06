from matplotlib import pyplot as plt

import numpy as np

from pytorch_arap.arap import ARAP_from_meshes, add_one_ring_neighbours,add_n_ring_neighbours
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.utils import ico_sphere
import os
import torch

from pytorch_arap.arap_utils import save_animation, plot_meshes

if torch.cuda.is_available():
	device = "cuda"
else:
	device = "cpu"

def deform_cuboid():
	targ = os.path.join("sample_meshes", "cuboid_hp.obj")
	meshes = load_objs_as_meshes([targ], load_textures=False)

	# meshes = ico_sphere(3)

	meshes = ARAP_from_meshes(meshes)  # convert to ARAP obejct

	# for cuboid
	O = meshes.verts_packed()[[22, 25]].mean(dim=0) # set origin to centre of mass
	nverts = meshes.num_verts_per_mesh()[0]
	meshes.offset_verts_(-O.unsqueeze(0).repeat(nverts, 1))  # normalise so [22] is on origin

	static_verts = [1,3,5,7,8,9,16,19,22,26,27,34,37,38,39,46,49,58,59,60,61,83,84,85,94,98,99,106,109,110,111,118,121,130,131,132,133,146,147,154,157,158,159,166,169,178,179,180,181,230,231,232,233,234,235,236,237,238,239,240,241,274,275,276,277,299,300,301,326,327,328,329,330,331,332,333,334,371,372,373,382]
	handle_verts = [0,2,4,6,14,15,17,18,25,32,33,35,36,44,45,47,48,70,71,72,73,74,75,76,97,104,105,107,108,116,117,119,120,142,143,144,145,152,153,155,156,164,165,167,168,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,286,287,288,289,290,291,292,353,354,355,356,357,358,359,360,361,362,363,364,385]

	## add neighbours for further constraint
	static_verts = add_one_ring_neighbours(meshes, static_verts)
	handle_verts = add_one_ring_neighbours(meshes, handle_verts)

	handle_pos = meshes.verts_padded()[0][handle_verts].clone()

	## plot mesh
	plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts, color="gray",
				change_lims=True)

	trisurfs, scatters = [], []

	n_frames = 90
	def anim(i):
		[x.remove() for x in trisurfs] # remove previous frame's mesh

		# 3D rotation matrix - theta deg about Z
		theta = (np.pi/4) * min(i/(n_frames/2), 1)
		s, c = np.sin(theta), np.cos(theta)
		R = torch.FloatTensor([ [c, -s, 0], [s, c, 0], [0, 0, 1] ])
		handle_pos_shifted = torch.mm(R, handle_pos.T).T
		handle_pos_shifted = handle_pos_shifted

		if i>n_frames/2:
			# shift in z
			handle_pos_shifted[:, 2] += (1+i-n_frames/2)*.07

		## deform, replot
		meshes.solve(static_verts=static_verts, handle_verts=handle_verts, handle_verts_pos=handle_pos_shifted,
					 n_its=0)  ## run ARAP
		trisurfs[:] = plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts)

	ax.axis("off")
	# anim(1)
	# plt.show()


	save_animation(fig, anim, n_frames=n_frames, title="cuboid_rotate")

def deform_sphere():

	meshes = ico_sphere(3)
	meshes = ARAP_from_meshes(meshes) # convert to ARAP obejct
	N = meshes.num_verts_per_mesh()[0]

	handle_verts = [26]
	handle_pos = meshes.verts_padded()[0][handle_verts]
	handle_pos_shifted = handle_pos.clone()

	# static as furthest vert
	static_verts = [max(range(N), key=lambda x: torch.norm(meshes.verts_padded()[0][x] - handle_pos[0]))]
	static_verts = add_n_ring_neighbours(meshes, static_verts, n = 5)

	trisurfs = plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts, prop=False, change_lims=True,
						   color="gray")

	disp_vec = meshes.C[0] - handle_pos[0]  # displace towards centre of mass

	n_frames = 100
	disp_frac = 1.2  # fraction of full disp_vec to move in animation
	step = disp_frac * 4/n_frames # moves

	def anim(i):
		[x.remove() for x in trisurfs] # remove previous frame's mesh

		if i < n_frames / 4 or i > 3 * n_frames / 4:
			direction = 1
		else:
			direction = -1

		handle_pos_shifted[0] += direction * step * disp_vec

		## deform, replot
		meshes.solve(static_verts=static_verts, handle_verts=handle_verts, handle_verts_pos=handle_pos_shifted, n_its = 1) ## run ARAP

		trisurfs[:] = plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts, prop=False,
								  color="gray")

	ax.axis("off")
	save_animation(fig, anim, n_frames=n_frames, title="sphere", fps = 30)

def deform_cactus():

	targ = os.path.join("sample_meshes", "cactus.obj")
	meshes = load_objs_as_meshes([targ], load_textures=False)
	meshes = ARAP_from_meshes(meshes, device=device) # convert to ARAP obejct
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
	trisurfs = plot_meshes(ax, meshes.verts_list(), faces, handle_verts=handle_verts, static_verts=static_verts, prop=prop, change_lims=True,
						   color="gray", zoom=1.5)

	disp_vec = torch.FloatTensor([1, 0, 0])  # displace in x direction

	n_frames = 40
	disp_frac = 0.4 # fraction of full disp_vec to move in animation
	step = disp_frac * 4/n_frames # moves

	nits = 2
	def anim(i):
		[x.remove() for x in trisurfs] # remove previous frame's mesh

		if i < n_frames / 4 or i > 3 * n_frames / 4:
			direction = 1
		else:
			direction = -1

		handle_pos_shifted[:] += direction * step * disp_vec

		## deform, replot
		verts = meshes.solve(static_verts=static_verts, handle_verts=handle_verts, handle_verts_pos=handle_pos_shifted, n_its = nits,
							 track_energy=False) ## run ARAP

		verts = [verts]

		trisurfs[:] = plot_meshes(ax, verts, faces, handle_verts=handle_verts, static_verts=static_verts, prop=prop,
								  color="gray")

	[anim(i) for i in range(1)]
	# anim(0)
	# f = 1
	# n_unknown = N - len(static_verts) - len(handle_verts)
	# [anim(i) for i in range(f)]

	# print(meshes.timer.report(nits=f, rots=f*nits, b1=f*nits, b2=f*nits, b3=f*nits))
	# plt.show()
	#
	ax.axis("off")
	save_animation(fig, anim, n_frames=n_frames, title="cactus", fps = 30)

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

	def anim(i):
		[x.remove() for x in trisurfs] # remove previous frame's mesh

		if i < n_frames / 4 or i > 3 * n_frames / 4:
			direction = 1
		else:
			direction = -1

		handle_pos_shifted[:] += direction * step * disp_vec

		## deform, replot
		verts = meshes.solve(static_verts=static_verts, handle_verts=handle_verts, handle_verts_pos=handle_pos_shifted, n_its = 1,
					 track_energy=False) ## run ARAP

		verts = [verts]

		trisurfs[:] = plot_meshes(ax, verts, faces, handle_verts=handle_verts, static_verts=static_verts, prop=prop,
								  color="gray")

	#

	# [anim(1) for i in range(1)]
	# plt.show()

	ax.axis("off")
	save_animation(fig, anim, n_frames=n_frames, title="smal", fps = 30)

def deform_dog():
	targ = os.path.join("sample_meshes", "dog.obj")
	meshes = load_objs_as_meshes([targ], load_textures=False)

	# meshes = ico_sphere(3)

	meshes = ARAP_from_meshes(meshes)  # convert to ARAP obejct

	meshes.rotate(rot_x=np.pi/2)

	# meshes.offset_verts_(-O.unsqueeze(0).repeat(nverts, 1))  # normalise so [22] is on origin
	# O = meshes.verts_packed()[[22, 25]].mean(dim=0) # set origin to centre of mass
	# nverts = meshes.num_verts_per_mesh()[0]

	handle_verts = add_n_ring_neighbours(meshes, [266], mesh_idx=0, n=2)
	static_verts = add_n_ring_neighbours(meshes, [523, 1134, 471, 1085], mesh_idx=0, n=1)

	## add neighbours for further constraint
	static_verts = add_one_ring_neighbours(meshes, static_verts)
	handle_verts = add_one_ring_neighbours(meshes, handle_verts)

	handle_pos = meshes.verts_padded()[0][handle_verts]
	# shift in z
	handle_pos_shifted = handle_pos.clone()

	## plot mesh
	t, s = plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts, change_lims=True,
					   color="dodgerblue")
	# [x.remove() for x in t+s]  # remove original mesh

	trisurfs, scatters = [], []

	def anim(i):
		[x.remove() for x in trisurfs+scatters] # remove previous frame's mesh

		if i < 20:
			handle_pos_shifted[:, 2] += 0.01

		elif i < 45:
			handle_pos_shifted[:, 0] += 0.0075

		elif i < 75:
			handle_pos_shifted[:, 0] -= 0.0075

		## deform, replot
		meshes.solve(static_verts=static_verts, handle_verts=handle_verts, handle_verts_pos=handle_pos_shifted,
					 n_its=0)  ## run ARAP
		trisurfs[:], scatters[:] = plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts,
											   color="orange")

	ax.axis("off")
	# ax.view_init(azim=-60)
	# plt.show()

	save_animation(fig, anim, n_frames=75)

if __name__ == "__main__":

	fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

	# deform_cuboid()
	# deform_sphere()
	# deform_smal()
	deform_cactus()




