# V-BASTA: Image Inpainting via Spatially Adaptive Convolutional Dictionary Learning

[![Status](https://img.shields.io/badge/Status-Under_Review-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2021a%2B-orange.svg)](#)

This repository provides the official MATLAB implementation for the **Image Inpainting** task using the **V-BASTA** (Variable-metric Backtracking Alternating Spatially-aware Thresholding Algorithm) framework. 

This specific module contains the $l_1$-norm continuous relaxation pre-training script (`demo_VBASTA_L1_Inpainting.m`). It reconstructs missing local structures from incomplete observations by coupling masked data fidelity with spatially adaptive convolutional sparse representation.

## 🌟 Core Features

- **Mask-Aware Gradient Computation**: Effectively handles incomplete observations by calculating gradients and descent conditions exclusively on the observed valid pixels, avoiding contamination from corrupted regions.
- **Spatially Adaptive Variable Metrics**: Adapts the update geometry to heterogeneous convolutional interactions. This avoids the conservative global step sizes that often fail to effectively recover complex missing local structures.
- **Defensive Backtracking Mechanism**: Guarantees stable convergence for both the dictionary and the sparse coefficients under the Kurdyka-Lojasiewicz framework, even with highly incomplete data (e.g., 50% missing pixels).
- **Unbiased Hot-Start Packaging**: Generates, locks, and rigorously saves the random mask arrays alongside the pre-trained variables. This ensures a strictly identical experimental environment for the subsequent $l_0$-norm fine-tuning phase.

## 📁 Repository Structure

```text
V-BASTA_Inpainting_L1/
├── datasets/                               % Image datasets for various visual recovery tasks
│   ├── city_100_100/
│   ├── denoisy/
│   ├── fruit_100_100/
│   ├── separation/
│   ├── standard/
│   └── standard1/                          % Target dataset for the inpainting task
├── image_helpers/                          % Image loading and preprocessing tools
│   ├── contrast_normalization/
│   ├── CreateImages.m / check_imgs_path.m / ...
│   └── split_folders_files.m
├── util/                                   % Low-level operators and compiled MEX files
│   ├── col2imstep.c / im2colstep.c         % C source files for fast patch operations
│   ├── col2imstep.mexw64 / im2colstep.mexw64 % Compiled MEX binaries
│   ├── vl_imarray.m / vl_imarraysc.m       % Visualization utilities
│   └── ...
├── compute_descent_lemma_upper_bound_VM.m  % Quadratic majorizer evaluation under variable metric
├── compute_H_gradients_in_parallel_mask.m  % Parallel batch gradient aggregator (Mask-Aware)
├── compute_patch_model_gradients_mask.m    % Time-domain gradient and residual calculation (Mask-Aware)
├── mycol2im.m                              % Image patch synthesis wrapper
├── myim2col.m                              % Image patch extraction wrapper
├── prox_vmetric_L2_ball.m                  % Exact Lagrangian projection for dictionary atoms
├── proxomiga2_2D.m                         % Fast L2 normalization for dictionary initialization
├── psnr.m                                  % Peak Signal-to-Noise Ratio evaluation
├── showDictionary.m                        % Dictionary visualization helper
├── soft_threshold.m                        % Standard L1 soft-thresholding operator
├── demo_VBASTA_L1_Inpainting.m             % Main entry script for L1 Image Inpainting
└── README.md                               % Project documentation

##Prerequisites
MATLAB (Tested on R2024a and later versions).
Image Processing Toolbox (Required for structural similarity ssim evaluation).
Parallel Computing Toolbox (Optional, but highly recommended for parfor acceleration during multi-image training).
Ensure that the compiled .mex files in the util/ folder are compatible with your operating system.

##Reproducibility Notes
To ensure full reproducibility of the experimental results presented in the manuscript:
The generated masks and corrupted images are strictly saved in the output .mat file.
You must run this $l_1$ pre-training script first before executing the $l_0$ inpainting script, as the latter relies on this exact saved environment to perform an unbiased hot-start evaluation.