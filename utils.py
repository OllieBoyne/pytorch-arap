import torch
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from tqdm import tqdm
from matplotlib.tri import Triangulation
from matplotlib.colors import ListedColormap
import os

def is_positive_definite(tensor):
	"""Bool check if 2D matrix is positive definite

	:type tensor: torch.Tensor"""

	eig, _ = torch.eig(tensor)

	eig = eig[:, 0] # only real part

	return torch.all(eig > 0)

def least_sq_with_known_values(A, b, known = None):
	"""Solves the least squares problem minx ||Ax - b||2, where some values of x are known.
	Works by moving all known variables from A to b.

	:param A: full rank matrix of size (mxn)
	:param b: matrix of size (m x k)
	:param known: dict of known_variable : value

	:type A: torch.Tensor
	:type B: torch.Tensor
	:type known: dict

	:returns x: matrix of size (n x k)
	"""

	M, N = A.shape
	M2, K = b.shape
	assert M == M2, "A's first dimension must match b's second"

	if known is None: known = {}

	# Move to b
	for index, val in known.items():
		col = A[:, index]
		b -= torch.einsum("i,j->ij", col, val)

	# Remove from A
	unknown = [n for n in range(N) if n not in known]
	A = A[:, unknown] # only continue with cols for unknowns

	x, QR = torch.lstsq(b, A)

	# all unknown values have now been found. Now create the output tensor, of known and unknown values in correct positions
	x_out = torch.zeros((N, K))
	if known is not None:
		# Assign initially known values to x_out
		for index, val in known.items():
			x_out[index] = val

		# Assign initially unknown values to x_out
		x_out[unknown] = x[:len(unknown)]

	## X has shape max(m, n) x k. Only want first n rows
	return x_out

from time import perf_counter as tfunc
class Timer:
	def __init__(self):
		self.t0 = tfunc()
		self.log = {}

	def add(self, label):
		"""nrepeats: optional value to multiply each value by.
		Either int, or iterable with valid length
		Used for timing the total time for an entire loop -
		nrepeats is length of iterator."""

		if label not in self.log: self.log[label] = []
		self.log[label] += [tfunc() - self.t0]
		self.t0 = tfunc()

	def report(self, nits=None, **custom_nits):
		"""Print report of log.
		if nits is none, assume the mean time for each operation is required.
		if nits is an int, divide the total time by nits
		any nits that differ can be given in custom_nits"""

		out = {}
		for k, t in self.log.items():
			if nits is None:
				out_time = np.mean(t)
			elif isinstance(nits, int):
				if k in custom_nits:
					out_time = np.sum(t) / custom_nits[k]
				else:
					out_time = np.sum(t) / nits

			out[k] = {"out_time":out_time, "mean":np.mean(t), "call_count": len(t)}

		return "\n".join([f"{k} = {t['out_time']*1000:.1f}ms [{t['call_count']} calls, {t['mean']*1e6:.1f}us/call]" for k, t in out.items()])


def simplify_obj_file(src):
	"""Given a .obj file, saves a copy with '_simplify' in fname,
	with everything except for vertices and faces."""

	out = []
	nverts = 3889
	with open(src) as infile:
		for line in infile.readlines():
			if line.startswith("v ") or line.startswith("o "):
				out += [line]

			elif line.startswith("f "):
				## for f, only preserve v, not vn or vt
				faceverts = [i.split("/")[0] for i in line.split()[1:]]
				out_line = "f " + " ".join(faceverts) + "\n"
				out += [out_line]

				# faceverts_n = list(map(int, faceverts))
				# if any(v>=nverts for v in faceverts_n):
				# 	print(line, "Faces have invalid indices")

	out_src = src.replace(".obj", "_simplify.obj")
	with open(out_src, "w") as outfile:
		outfile.writelines(out)

## matplotlib utils

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

def plot_meshes(ax, verts, faces, static_verts=[], handle_verts=[], change_lims=False, color="darkcyan",
				prop=True, zoom=1.5, n_meshes=1):
	"""
	:type mesh: ARAPMeshes
	:type rots: array to prerotate a mesh by
	"""

	trisurfs = []

	for n in range(n_meshes):
		points = verts[n].detach().numpy()
		faces = faces[n].detach().numpy()

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

def save_animation(fig, func, n_frames, fmt="gif", fps=15, title="output", **kwargs):
	"""Save matplotlib animation."""

	writer = writers['imagemagick']
	W = writer(fps = fps, bitrate=1500)

	anim = FuncAnimation(fig, func, frames=n_frames, **kwargs)

	with tqdm(total=n_frames) as save_progress:
		anim.save(os.path.join("animations", f"{title}.{fmt}"), writer=W,
					   progress_callback=lambda x, i: save_progress.update())

if __name__ == "__main__":

	simplify_obj_file("sample_meshes/smal.obj")

