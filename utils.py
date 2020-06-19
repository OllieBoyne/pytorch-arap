import torch
from matplotlib import pyplot as plt
import numpy as np

def least_sq_with_known_values(A, b, known = None):
	"""Solves the least squares problem minx ||Ax - b||2, where some values of x are known.
	Works by moving all known variables from A to b.

	:param known: dict of known_variable : value
	:param A: full rank matrix of size (mxn)
	:param b: matrix of size (m x k)

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
	with open(src) as infile:
		for line in infile:
			pass

if __name__ == "__main__":

	# test lstsq with known values
	y = [1, 3.5, 4]
	x = torch.FloatTensor( [2, 2.5, 3] )
	A = torch.ones((len(x), 2))
	A[:, 0] = x

	b = torch.FloatTensor( y ).unsqueeze(-1)

	# standard regression
	regr = least_sq_with_known_values(A, b)
	m, c = regr

	# regression with c = 0 fixed
	regr_mod = least_sq_with_known_values(A, b, known={1:torch.FloatTensor([0])})
	m_mod, c_mod = regr_mod

	plt.scatter(x, y)
	xrange = np.arange(0, x.max(), 0.01)
	plt.plot(xrange, m * xrange + c, ls="--")
	plt.plot(xrange, m_mod * xrange + c_mod)

	plt.show()
