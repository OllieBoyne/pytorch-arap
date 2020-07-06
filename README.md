# Pytorch ARAP

This module implements energy calculations and iterative solvers for the As-Rigid-As-Possible Surface Modelling method (Sorkine & Alexa, 2007).

Builds on PyTorch and PyTorch3D.

## Installation

1. Clone into target repository
2. **Optional**: Install torch_batch_svd, from https://github.com/KinglittleQ/torch-batch-svd for improved SVD calculations. If this is not installed, torch.svd will be used.

## Usage
For examples on how to use this module, see `demo_loss.py` for calculating the ARAP surface energy for as a loss function,
and `demo_solver.py` for a full solver
