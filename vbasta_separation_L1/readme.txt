# V-BASTA: Cartoon-Texture Separation via Spatially Adaptive Convolutional Dictionary Learning

This repository provides the official MATLAB implementation for the **Cartoon-Texture Separation** task using the **V-BASTA** (Variable-metric Backtracking Alternating Spatially-aware Thresholding Algorithm) framework. 

This specific module contains the $l_1$-norm continuous relaxation pre-training script (`demo_VBASTA_L1_Separation.m`). It performs structural decomposition by coupling spatially adaptive convolutional sparse representation (for the oscillatory texture component) with Total Variation (TV) regularization (for the piecewise-smooth cartoon component).

## 🌟 Core Features

- **Structural Decomposition**: Elegantly separates heterogeneous visual content. The texture is modeled via V-BASTA's time-domain convolutional dictionary learning, while the cartoon part is extracted using TV denoising.
- **Spatially Adaptive Variable Metrics**: Accelerates the learning of texture representations by aligning the update geometry with position-dependent convolutional overlap structures.
- **Intelligent Best Tracking**: Independently tracks the highest PSNR for both cartoon and texture components, dynamically saving the optimal decomposition state.
- **Early Stopping Mechanism**: Implements a patience-based early stopping criterion to halt iterations when no further structural improvements are observed, ensuring computational efficiency.

## 📁 Repository Structure

```text
V-BASTA_Separation_L1/
├── datasets/                               % Image datasets for various visual recovery tasks
│   ├── city_100_100/
│   ├── denoisy/
│   ├── fruit_100_100/
│   ├── separation/                         % Contains composite source data (e.g., cat4.mat, fense2.mat)
│   ├── standard/
│   └── standard1/
├── deconvtv_v1/                            % Total Variation (TV) Denoising library for Cartoon extraction
│   ├── data/
│   ├── private/
│   ├── deconvtv.m / deconvtvl1.m / ...     % Core TV solvers
│   └── user_guide.pdf
├── image_helpers/                          % Image loading and preprocessing tools
│   ├── contrast_normalization/
│   ├── CreateImages.m / split_folders_files.m / ...
│   └── ...
├── util/                                   % Low-level operators and compiled MEX files
├── compute_descent_lemma_upper_bound_VM.m  % Quadratic majorizer evaluation under variable metric
├── compute_H_gradients_in_parallel.m       % Parallel batch gradient aggregator
├── compute_patch_model_gradients.m         % Time-domain gradient and residual calculation
├── mycol2im.m                              % Image patch synthesis wrapper
├── myim2col.m                              % Image patch extraction wrapper
├── prox_vmetric_L2_ball.m                  % Exact Lagrangian projection for dictionary atoms
├── proxomiga2_2D.m                         % Fast L2 normalization for dictionary initialization
├── psnr.m                                  % Peak Signal-to-Noise Ratio evaluation
├── showDictionary.m                        % Dictionary visualization helper
├── soft_threshold.m                        % Standard L1 soft-thresholding operator
├── demo_VBASTA_L1_Separation.m             % Main entry script for L1 Cartoon-Texture Separation
└── README.md                               % Project documentation

##Reproducibility Notes
To ensure full reproducibility of the experimental results presented in the manuscript:
The optimal model tracking excludes the 1st-10th iteration to avoid logging artificial spikes caused by initialization shock.
The output .mat file (vbasta_l1_separation_results.mat) is specifically packaged to provide an unbiased hot-start environment for the subsequent strict $l_0$-norm fine-tuning phase.