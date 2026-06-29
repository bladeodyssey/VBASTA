%% Variable-metric Backtracking Alternating Spatially-aware Thresholding Algorithm (V-BASTA) for L1 Image Inpainting
% =========================================================================
% Features:
% 1. Executes L1 soft-thresholding dictionary learning pre-training for inpainting.
% 2. Strictly generates and locks random masks and degraded images.
% 3. Packages and saves D_L, alpha_cells, and the exact mask environment to provide
%    an unbiased hot-start environment for the subsequent L0 phase.
%revised 2026.6.17 Minghui Xu
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. Parameter Settings
fprintf('--- 1. Parameter Settings (V-BASTA L1 Inpainting Pre-training) ---\n');
m = 100;              % Number of dictionary atoms
patch_size = 11;      % Spatial filter size
lambda1 = 0.04;       % L1 sparsity regularization parameter
noiseSD = 0;          % Initial noise standard deviation

MAXITER = 300;        % Maximum pre-training iterations
max_backtrack_iter = 50; 
gamma1 = 0.006;         
gamma2 = 0.006;         
tau1 = 1.5;           % Backtracking decay multiplier for alpha
tau2 = 1.5;           % Backtracking decay multiplier for D_L

%% 2. Data Loading and 50% Missing Mask Generation
fprintf('--- 2. Loading Data and Generating 50%% Masks ---\n');
try 
    L_data = CreateImages('datasets/standard1', 'none', 0, 'gray'); 
    y_clean_cells = squeeze(num2cell(L_data, [1, 2, 3])); 
catch
    warning('Loading failed. Using randomly generated test data instead.'); 
    for i=1:2, y_clean_cells{i} = rand(100, 100); end 
end
numpic = length(y_clean_cells);

M_cells = cell(1, numpic);
y_masked_cells = cell(1, numpic);
image_sizes = cell(1, numpic); 
N_l_vec = zeros(1, numpic); 

% Generate masks and corrupted images (Crucial: These must be saved for L0 hot-start)
for l = 1:numpic
    sz = [size(y_clean_cells{l}, 1), size(y_clean_cells{l}, 2)];
    image_sizes{l} = sz;
    
    M_cells{l} = zeros(sz);
    M_cells{l}(rand(sz) < 0.5) = 1; % 50% random missing mask
    
    y_noisy = y_clean_cells{l} + noiseSD * randn(sz);
    y_masked_cells{l} = M_cells{l} .* y_noisy;
    
    dummy_patches = myim2col(zeros(sz), patch_size); 
    N_l_vec(l) = size(dummy_patches, 2); 
end
fprintf('Preprocessing complete. Loaded %d images.\n\n', numpic);

%% 3. Initialization and Memory Cache
fprintf('--- 3. Initializing Dictionary, Sparse Codes, and Memory Cache ---\n');
d = patch_size^2;
D_L = randn(d, m);
D_L = proxomiga2_2D(D_L); 
alpha_cells = cell(1, numpic);
for l = 1:numpic
    alpha_cells{l} = zeros(m, N_l_vec(l)); 
end

tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

history.total_objective = zeros(1, MAXITER);
history.data_error = zeros(1, MAXITER);
history.l1_penalty = zeros(1, MAXITER);
history.sparsity = zeros(1, MAXITER);
history.time = zeros(1, MAXITER); 
history.psnr = zeros(1, MAXITER);

%% 4. V-BASTA Inpainting Main Loop (L1 Soft Thresholding)
fprintf('--- 4. Starting V-BASTA L1 Inpainting Pre-training Loop ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: Update Sparse Coefficients (alpha) via VM + Soft Thresholding + Mask
    % ==========================================================
    abs_D = abs(D_L); D_sum = sum(abs_D, 2); Dt = D_L'; 
    
    % --- Intelligent Caching for S_alpha ---
    S_alpha_cached_list = cell(1, numpic);
    unique_sizes = []; unique_S_alpha = {};
    for l = 1:numpic
        sz = image_sizes{l};
        idx = [];
        if ~isempty(unique_sizes)
            matches = (unique_sizes(:,1) == sz(1)) & (unique_sizes(:,2) == sz(2));
            idx = find(matches, 1);
        end
        if isempty(idx)
            D_rep = repmat(D_sum, 1, N_l_vec(l));
            Img_overlap = mycol2im(D_rep, sz, patch_size);
            P_overlap = myim2col(Img_overlap, patch_size);
            new_S = abs_D' * P_overlap + 1e-8;
            unique_sizes = [unique_sizes; sz]; 
            unique_S_alpha{end+1} = new_S;     
            idx = length(unique_S_alpha);
        end
        S_alpha_cached_list{l} = unique_S_alpha{idx};
    end
    
    new_alpha_cells = cell(1, numpic);
    
    parfor l = 1:numpic
        old_alpha_l = alpha_cells{l}; 
        y_masked_l = y_masked_cells{l}; 
        M_l = M_cells{l};
        img_size_l = image_sizes{l}; 
        S_alpha_l = S_alpha_cached_list{l};
        
        [grad_alpha_l, ~, H_old] = compute_patch_model_gradients_mask(D_L, Dt, old_alpha_l, y_masked_l, M_l, img_size_l, patch_size);
        
        tau1_curr = max(gamma1, tau1_history(l) / 1.2); 
        alpha_l_candidate = old_alpha_l; 
        
        for k_bt = 1:max_backtrack_iter
            S_effective = tau1_curr * S_alpha_l; 
            v_alpha = old_alpha_l - grad_alpha_l ./ S_effective;
            
            % L1 Soft Thresholding
            lambda_th = lambda1 ./ S_effective; 
            alpha_l_candidate = soft_threshold(v_alpha, lambda_th);
            
            y_recon_new = mycol2im(D_L * alpha_l_candidate, img_size_l, patch_size);
            f_val = 0.5 * sum((M_l .* (y_recon_new - y_masked_l)).^2, 'all');
            Q_val = H_old + compute_descent_lemma_upper_bound_VM(old_alpha_l, alpha_l_candidate, grad_alpha_l, S_effective);
            
            if f_val <= Q_val
                tau1_history(l) = tau1_curr; 
                break; 
            else
                tau1_curr = tau1_curr * tau1; 
            end
        end
        
        % Failure handling for sparse coefficients update
        if k_bt == max_backtrack_iter
            warning('Maximum backtracking steps reached for sample %d. Descent condition not strictly met.', l);
        end
        
        new_alpha_cells{l} = alpha_l_candidate;
    end
    alpha_cells = new_alpha_cells; 
    
    % ==========================================================
    % Phase 2: Update Dictionary (D_L) via Variable-Metric Projection + Mask
    % ==========================================================
    [~, grad_D_H_total, H_old_D_update] = compute_H_gradients_in_parallel_mask(D_L, alpha_cells, y_masked_cells, M_cells, image_sizes, patch_size);
    
    S_D = zeros(d, m);
    for l = 1:numpic
        abs_alpha_l = abs(alpha_cells{l});
        alpha_sum = sum(abs_alpha_l, 1); 
        X = repmat(alpha_sum, d, 1); 
        Img_overlap = mycol2im(X, image_sizes{l}, patch_size); 
        P_overlap = myim2col(Img_overlap, patch_size);         
        S_D = S_D + P_overlap * abs_alpha_l';                      
    end
    S_D = S_D + 1e-8; 
    
    old_D_L = D_L; 
    tau2_curr = max(gamma2, tau2_history / 1.2);
    
    for k_bt = 1:max_backtrack_iter
        S_effective_D = tau2_curr * S_D;
        
        v_D = old_D_L - grad_D_H_total ./ S_effective_D; 
        D_candidate = prox_vmetric_L2_ball(v_D, S_effective_D);
        
        H_new_D_update = 0;
        for l = 1:numpic
            y_recon_l = mycol2im(D_candidate * alpha_cells{l}, image_sizes{l}, patch_size); 
            H_new_D_update = H_new_D_update + 0.5 * sum((M_cells{l} .* (y_recon_l - y_masked_cells{l})).^2, 'all'); 
        end
        
        Q_val = H_old_D_update + compute_descent_lemma_upper_bound_VM(old_D_L, D_candidate, grad_D_H_total, S_effective_D);
        if H_new_D_update <= Q_val
            tau2_history = tau2_curr; 
            break; 
        else
            tau2_curr = tau2_curr * tau2; 
        end
    end
    
    % Failure handling for dictionary update
    if k_bt == max_backtrack_iter
        warning('Maximum backtracking steps reached for Dictionary update.');
    end
    
    D_L = D_candidate; 
    
    % ==========================================================
    % Phase 3: Logging and Evaluation
    % ==========================================================
    iter_time = toc;
    if outerIter == 1
        history.time(outerIter) = iter_time;
    else
        history.time(outerIter) = history.time(outerIter-1) + iter_time;
    end
    
    data_error_term = H_new_D_update; 
    l1_penalty_term = lambda1 * sum(abs(cell2mat(alpha_cells)), 'all');
    total_objective = data_error_term + l1_penalty_term;

    history.total_objective(outerIter) = total_objective;
    history.data_error(outerIter) = data_error_term;
    history.l1_penalty(outerIter) = l1_penalty_term;
    history.sparsity(outerIter) = sum(cellfun(@nnz, alpha_cells)) / (sum(N_l_vec)*m);
    
    current_psnr = 0;
    for l = 1:numpic
        y_recon_curr = mycol2im(D_L * alpha_cells{l}, image_sizes{l}, patch_size);
        y_final_curr = M_cells{l} .* y_clean_cells{l} + (1 - M_cells{l}) .* y_recon_curr;
        current_psnr = current_psnr + psnr(y_clean_cells{l}, y_final_curr);
    end
    history.psnr(outerIter) = current_psnr / numpic;
    
    fprintf('Iter %04d: Obj=%.3e, Sparsity=%.3f%%, PSNR=%.2fdB, Time=%.2fs\n', ...
            outerIter, total_objective, history.sparsity(outerIter)*100, history.psnr(outerIter), history.time(outerIter));
end

%% 5. Save Results (Crucial for Hot-Start)
SAVE_FILE = 'vbasta_l1_inpainting_results.mat';
fprintf('\n--- 5. Saving L1 Pre-training Results and Mask Environments to %s ---\n', SAVE_FILE);
% Strict packaging: masks and images are saved along with optimized variables to ensure rigorous L0 hot-start comparison
save(SAVE_FILE, 'D_L', 'alpha_cells', 'history', ...
     'y_clean_cells', 'y_masked_cells', 'M_cells', 'image_sizes', 'N_l_vec');
fprintf('Results successfully saved. The L0 script can directly load this file for exact benchmarking.\n\n');

%% 6. Final Evaluation and Visualization
fprintf('--- 6. Final Visualization of L1 Pre-training ---\n');
figure('Name', 'V-BASTA L1 Inpainting - Convergence');
plot(1:MAXITER, history.total_objective, 'b-', 'LineWidth', 2);
xlabel('Iterations'); ylabel('Total Objective Value'); title('L1 Pre-training Convergence Curve');
grid on; set(gca, 'YScale', 'log');

img_idx_to_show = 1;
y_original = y_clean_cells{img_idx_to_show};
y_masked = y_masked_cells{img_idx_to_show};
M_mask = M_cells{img_idx_to_show};
alpha_final = alpha_cells{img_idx_to_show};

y_reconstructed = mycol2im(D_L * alpha_final, image_sizes{img_idx_to_show}, patch_size);
y_final = M_mask .* y_original + (1 - M_mask) .* y_reconstructed;
final_psnr = psnr(y_original, y_final);

figure('Name', 'V-BASTA L1 Inpainting - Visual Results', 'Position', [100, 100, 1200, 400]);
subplot(1, 4, 1); imshow(y_original, []); title('Original Image');
subplot(1, 4, 2); imshow(y_masked, []); title('Degraded Image (50% Missing)');
subplot(1, 4, 3); imshow(y_reconstructed, []); title('Pure Dictionary L1 Reconstruction');
subplot(1, 4, 4); imshow(y_final, []); title(sprintf('L1 Inpainting Result\n(PSNR: %.2f dB)', final_psnr));