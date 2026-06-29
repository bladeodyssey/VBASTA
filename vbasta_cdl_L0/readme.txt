# V-BASTA: Variable-metric Backtracking Alternating Spatially-aware Thresholding Algorithm (L0-Norm)

[![Status](https://img.shields.io/badge/Status-Under_Review-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2021a%2B-orange.svg)](#)

This repository provides the official MATLAB implementation of the **strict $l_0$-norm** formulation of **V-BASTA**, a time-domain convolutional dictionary learning (CDL) framework designed for learning compact, interpretable visual representations under spatially heterogeneous local interactions.

This specific module (`main_VBASTA_L0.m`) implements the non-convex $l_0$-norm optimization phase, featuring spatially-aware dynamic hard thresholding and a robust hot-start mechanism to transition smoothly from continuous relaxations.

## 🌟 Core Features

- **Strict $l_0$-Norm Constraints via Dynamic Hard Thresholding**: Utilizes the exact $l_0$ proximal operator under a variable metric. The truncation threshold dynamically adapts to the local convolutional overlap density, preventing premature support freezing.
- **Robust Hot-Start Mechanism**: Seamlessly loads pre-trained variables from the $l_1$ relaxation phase, effectively avoiding the "thresholding shock" typically associated with non-convex strict sparsity initialization.
- **Dual-Objective Logging**: Simultaneously evaluates and visualizes the internal optimization-driven objective (Data Fidelity + $\lambda ||\alpha||_0$) and the standard reporting objective (Data Fidelity + Exact Non-zero Count) for rigorous paper benchmarking.
- **Spatially Adaptive Variable Metrics**: Abandons globally uniform scalar update rules, utilizing diagonal majorizers to perfectly align the update geometry with position-dependent spatial structures.

## 📁 Repository Structure

```text
V-BASTA_CDL_L0/
├── datasets/                               % Image datasets for evaluation
│   ├── city_100_100/                       
│   ├── fruit_100_100/                      
│   └── ...
├── image_helpers/                          % Image loading and preprocessing tools
│   ├── CreateImages.m                      % Core script to load and format image datasets
│   └── ...
├── util/                                   % Low-level operators and visualization utilities
│   ├── col2imstep.mexw64 / im2colstep.mexw64 % Compiled MEX files for fast patch operations
│   ├── vl_imarray.m / vl_imarraysc.m       % Dictionary visualization helpers
│   └── ...
├── compute_descent_lemma_upper_bound_VM.m  % Quadratic majorizer evaluation under variable metric
├── compute_H_gradients_in_parallel.m       % Parallel batch gradient aggregator
├── compute_patch_model_gradients.m         % Time-domain gradient and residual calculation
├── mycol2im.m                              % Image patch synthesis wrapper
├── myim2col.m                              % Image patch extraction wrapper
├── prox_vmetric_L2_ball.m                  % Exact Newton's projection for variable-metric L2 ball
├── psnr.m                                  % Peak Signal-to-Noise Ratio evaluation
├── showDictionary.m                        % Dictionary visualizer
├── main_VBASTA_L0.m                        % Main entry script for V-BASTA L0 optimization
└── README.md                               % Project documentation

##Prerequisites
MATLAB (Tested on R2024a and later versions).

Image Processing Toolbox (Required for structural similarity ssim evaluation).

Parallel Computing Toolbox (Optional, but highly recommended for parfor acceleration during multi-image training).

Ensure that the compiled .mex files in the util/ folder are compatible with your operating system.

Pre-trained L1 Results: A saved .mat file from the V-BASTA L1 phase is required if the USE_HOT_START flag is enabled.

##Reproducibility Notes
To ensure full reproducibility of the experimental results presented in the manuscript:

The internal sparsity controller lambda0 acts as the thresholding regulator. Adjusting this parameter directly controls the trade-off between representation compactness and reconstruction fidelity.

If the hot-start verification fails (e.g., due to dimension mismatch), the code will automatically issue a warning and gracefully fall back to random initialization with strict spherical projection.