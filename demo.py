from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation, writers
from matplotlib.colors import ListedColormap
import numpy as np

from solver import ARAPMeshes, ARAP_from_meshes, add_one_ring_neighbours,add_n_ring_neighbours
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.utils import ico_sphere
import os
import torch

from tqdm import tqdm

def save_animation(fig, func, n_frames, fmt="gif", fps=15, title="output", **kwargs):

	writer = writers['imagemagick']
	W = writer(fps = fps, bitrate=1500)

	anim = FuncAnimation(fig, func, frames=n_frames, **kwargs)

	with tqdm(total=n_frames) as save_progress:
		anim.save(os.path.join("animations", f"{title}.{fmt}"), writer=W,
					   progress_callback=lambda x, i: save_progress.update())

def equal_3d_axes(ax, X, Y, Z, zoom=1.0):

	"""Sets all axes to same lengthscale through trick found here:
	https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to"""

	xmax, xmin, ymax, ymin, zmax, zmin = X.max(), X.min(), Y.max(), Y.min(), Z.max(), Z.min()

	max_range = np.array([xmax - xmin, ymax - ymin, zmax - zmin]).max() / (2.0 * zoom)

	mid_x = (xmax + xmin) * 0.5
	mid_y = (ymax + ymin) * 0.5
	mid_z = (zmax + zmin) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)

def plot_meshes(ax, meshes, static_verts=[], handle_verts=[], change_lims=False, color="darkcyan",
				prop=True, zoom=1.5):
	"""
	:type mesh: ARAPMeshes
	:type rots: array to prerotate a mesh by
	"""

	trisurfs = []

	for n in range(len(meshes)):
		points = meshes.verts_list()[n].detach().numpy()
		faces = meshes.faces_list()[n].detach().numpy()


		X, Y, Z = np.rollaxis(points, -1)
		tri = Triangulation(X, Y, triangles=faces).triangles

		cmap = ListedColormap([color, "black", "red"], "mesh")  # colourmap used for showing properties on mesh

		trisurf_shade = ax.plot_trisurf(X, Y, Z, triangles=tri, alpha=0.9, color=color, shade=True)  # shade entire mesh
		trisurfs += [trisurf_shade]
		if prop:
			trisurf_prop = ax.plot_trisurf(X, Y, Z, triangles=tri, alpha=0.5, cmap=cmap)  # display properties of faces
			trisurfs += [trisurf_prop]

		if prop:
			# Set colours based on handles
			vert_prop = np.zeros((len(X))) #property of each vert - handle, static or neither
			vert_prop[handle_verts] = 1
			vert_prop[static_verts] = 0.5

			colors = vert_prop[tri].max(axis=1) # facecolor based on maximum property of connecting verts
			trisurf_prop.set_array(colors)


	if change_lims: equal_3d_axes(ax, X, Y, Z, zoom=zoom)

	return trisurfs

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
		[x.remove() for x in trisurfs+scatters] # remove previous frame's mesh

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
		trisurfs[:], scatters[:] = plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts)

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

	targ = os.path.join("sample_meshes", "lp_cactus.obj")
	meshes = load_objs_as_meshes([targ], load_textures=False)
	meshes = ARAP_from_meshes(meshes) # convert to ARAP obejct
	N = meshes.num_verts_per_mesh()[0]

	meshes.rotate(mesh_idx=0, rot_x=np.pi/2)

	# handle as topmost vert
	handle_verts = [79]
	handle_verts = add_one_ring_neighbours(meshes, handle_verts)
	handle_pos = meshes.verts_padded()[0][handle_verts]
	handle_pos_shifted = handle_pos.clone()

	# static as base
	static_verts = [41] # centres of paws
	static_verts = add_n_ring_neighbours(meshes, static_verts, n = 1)

	prop = True
	trisurfs = plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts, prop=prop, change_lims=True,
						   color="gray", zoom=1.5)

	disp_vec = torch.FloatTensor([1, 0, 0])  # displace in x direction

	n_frames = 30
	disp_frac = 0.02 # fraction of full disp_vec to move in animation
	step = disp_frac * 4/n_frames # moves

	def anim(i):
		[x.remove() for x in trisurfs] # remove previous frame's mesh

		if i < n_frames / 4 or i > 3 * n_frames / 4:
			direction = 1
		else:
			direction = -1

		handle_pos_shifted[:] += direction * step * disp_vec

		## deform, replot
		meshes.solve(static_verts=static_verts, handle_verts=handle_verts, handle_verts_pos=handle_pos_shifted, n_its = 1,
					 track_energy=False) ## run ARAP

		trisurfs[:] = plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts, prop=prop,
								  color="gray")

	#

	# anim(1)
	plt.show()

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
	static_verts = add_n_ring_neighbours(meshes, static_verts, n = 5)
	static_verts = []


	prop = True
	trisurfs = plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts, prop=prop, change_lims=True,
						   color="gray", zoom=1.5)

	disp_vec = torch.FloatTensor([1, 0, 0])  # displace in x direction

	n_frames = 100
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
		meshes.solve(static_verts=static_verts, handle_verts=handle_verts, handle_verts_pos=handle_pos_shifted, n_its = 1,
					 track_energy=False) ## run ARAP

		trisurfs[:] = plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts, prop=prop,
								  color="gray")

	#

	[anim(1) for i in range(3)]
	plt.show()

	ax.axis("off")
	save_animation(fig, anim, n_frames=n_frames, title="cactus", fps = 30)

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

	# deform_sphere()
	deform_smal()
	# deform_cactus()



