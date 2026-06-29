# V-BASTA: Variable-metric Backtracking Alternating Spatially-aware Thresholding Algorithm

This repository provides the official MATLAB implementation of **V-BASTA**, a time-domain convolutional dictionary learning (CDL) framework that enforces locality-preserving strict sparsity and spatially adaptive learning geometry. 

This repository contains the $l_1$-norm continuous relaxation pre-training implementation (`main_VBASTA_L1.m`), which serves as a robust foundation for learning compact, interpretable visual representations under spatially heterogeneous local interactions.

## 🌟 Core Features

- **Time-Domain Locality Preservation**: Operates directly in the spatial domain to avoid the boundary effects and locality degradation commonly associated with Fourier-domain reformulations.
- **Spatially Adaptive Variable Metrics**: Abandons globally uniform scalar update rules. Instead, it utilizes diagonal majorizers (derived via the Gerschgorin Disc Theorem) to perfectly align the update geometry with position-dependent convolutional overlap structures.
- **Defensive Backtracking Mechanism**: Integrates a robust backtracking strategy to ensure stable updates and strict descent conditions, guaranteeing convergence to a critical point under the Kurdyka-Lojasiewicz framework.
- **High-Efficiency Caching**: Features deep memory caching for redundant local spatial operators and step-size "warm starts" to significantly reduce computational overhead.

## 📁 Repository Structure

```text
V-BASTA_CDL/
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
├── prox_vmetric_L2_ball.m                  % Exact Lagrangian projection for dictionary atoms
├── proxomiga2_2D.m                         % Fast L2 normalization for dictionary initialization
├── psnr.m                                  % Peak Signal-to-Noise Ratio evaluation
├── showDictionary.m                        % Dictionary visualizer
├── soft_threshold.m                        % Standard soft-thresholding operator
├── main_VBASTA_L1.m                        % Main entry script for V-BASTA L1 optimization
└── README.md                               % Project documentation

## Prerequisites
MATLAB (Tested on R2024a and later versions).

Image Processing Toolbox (Required for structural similarity ssim evaluation).

Parallel Computing Toolbox (Optional, but highly recommended for parfor acceleration during multi-image training).

Ensure that the compiled .mex files in the util/ folder are compatible with your operating system.

## Reproducibility Notes
To ensure full reproducibility of the experimental results presented in the manuscript:

The random seed can be fixed prior to dictionary initialization.

If the required maximum backtracking steps (max_backtrack_iter) are exhausted, the algorithm is designed to issue a protective warning and gracefully fall back to a conservative step size, preventing optimization collapse.