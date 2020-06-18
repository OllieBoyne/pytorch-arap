from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

from solver import ARAPMeshes, ARAP_from_meshes
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.utils import ico_sphere
import os
import torch


def equal_3d_axes(ax, X, Y, Z, zoom=1):

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

def plot_meshes(ax, meshes, static_verts=[], handle_verts=[]):
	"""
	:type mesh: ARAPMeshes
	"""

	trisurfs = []
	for mesh in meshes:
		points, faces = mesh.mesh_to_numpy()

		X, Y, Z = np.rollaxis(points, -1)
		tri = Triangulation(X, Y, triangles=faces)
		trisurf = ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, alpha=1.0)

		for v in static_verts:
			ax.scatter(X[v], Y[v], Z[v], color="red")

		for v in handle_verts:
			ax.scatter(X[v], Y[v], Z[v], color="orange")

	equal_3d_axes(ax, X, Y, Z)

	return trisurfs

if __name__ == "__main__":

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	targ = os.path.join("sample_meshes", "cuboid_hp.obj")
	meshes = load_objs_as_meshes([targ], load_textures=False)

	# meshes = ico_sphere(3)

	meshes = ARAP_from_meshes(meshes) # convert to ARAP obejct


	# for cactus
	# static_verts = [10] + meshes.one_ring_neighbours[0][10] # fix bottom middle, and all adjacent verts
	# handle_verts = [134] # top of cactus
	# handle_pos = meshes.verts_padded()[0][handle_verts[0]]
	# handle_pos[1] += 30
	# handle_pos = np.array([handle_pos.detach().numpy()])

	# for cuboid
	O = meshes.verts_packed()[[22, 25]].mean(dim=0) # set origin to centre of mass
	nverts = meshes.num_verts_per_mesh()[0]
	meshes.offset_verts_(-O.unsqueeze(0).repeat(nverts, 1))  # normalise so [22] is on origin

	static_verts = [1,3,5,7,8,9,16,19,22,26,27,34,37,38,39,46,49,58,59,60,61,83,84,85,94,98,99,106,109,110,111,118,121,130,131,132,133,146,147,154,157,158,159,166,169,178,179,180,181,230,231,232,233,234,235,236,237,238,239,240,241,274,275,276,277,299,300,301,326,327,328,329,330,331,332,333,334,371,372,373,382]
	handle_verts = [0,2,4,6,14,15,17,18,25,32,33,35,36,44,45,47,48,70,71,72,73,74,75,76,97,104,105,107,108,116,117,119,120,142,143,144,145,152,153,155,156,164,165,167,168,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,286,287,288,289,290,291,292,353,354,355,356,357,358,359,360,361,362,363,364,385]
	handle_pos = meshes.verts_padded()[0][handle_verts]

	## 3D rotation matrix - theta deg about Z
	theta = np.pi/2
	s, c = np.sin(theta), np.cos(theta)
	R = torch.FloatTensor([ [c, -s, 0], [s, c, 0], [0, 0, 1] ])
	handle_pos_rotated = torch.mm(R, handle_pos.T).T
	handle_pos_rotated = handle_pos_rotated.detach().numpy().copy()

	## plot mesh
	# plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts)
	## deform, replot
	meshes.solve(static_verts=static_verts, handle_verts=handle_verts, handle_verts_pos=handle_pos_rotated, n_its = 2) ## run ARAP
	plot_meshes(ax, meshes, handle_verts=handle_verts, static_verts=static_verts)
	plt.show()
