%% Variable-metric Backtracking Alternating Spatially-aware Thresholding Algorithm (V-BASTA) for L1-CDL
% =========================================================================
% Features:
% 1. Adaptive Metric: Utilizes Gerschgorin disc bounds as the spatially adaptive 
%    variable-metric matrices (S_alpha and S_D).
% 2. Deep Caching: Extracts redundant computations exploiting image dimension 
%    consistency, significantly reducing O(N) operator calls.
% 3. Warm Start: Caches historical step sizes for each image to skip invalid 
%    backtracking steps, accelerating convergence.
% 4. Defensive Backtracking: Includes failure handling to ensure optimization stability.
%revise 2026.6.17 Minghui Xu
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. Parameter Settings
fprintf('--- 1. Parameter Settings (V-BASTA L1 Pre-training) ---\n');
m = 100;              % Number of dictionary atoms
patch_size = 11;      % Spatial filter size
lambda1 = 1.2;        % L1 sparsity regularization parameter

% V-BASTA specific backtracking and metric scaling parameters
MAXITER = 300;       
max_backtrack_iter = 50; 
gamma1 = 0.006;       % Initial metric scaling parameter for S_alpha
gamma2 = 0.006;       % Initial metric scaling parameter for S_D
tau1 = 1.5;           % Backtracking decay multiplier for alpha update
tau2 = 1.5;           % Backtracking decay multiplier for D_L update

%% 2. Data Loading and Preprocessing
fprintf('--- 2. Loading and Preprocessing Data ---\n');
try 
    L_data = CreateImages('datasets\city_100_100', 'local_cn', 1, 'gray'); 
    y_cells = squeeze(num2cell(L_data, [1, 2, 3])); 
catch
    warning('Loading failed. Using randomly generated test data instead.'); 
    for i = 1:2, y_cells{i} = rand(100, 100); end 
end
numpic = length(y_cells);

image_sizes = cell(1, numpic); 
N_l_vec = zeros(1, numpic); 
for l = 1:numpic
    y_l = y_cells{l}; 
    image_sizes{l} = [size(y_l, 1), size(y_l, 2)]; 
    dummy_patches = myim2col(zeros(image_sizes{l}), patch_size); 
    N_l_vec(l) = size(dummy_patches, 2); 
end
fprintf('Preprocessing complete. Loaded %d images.\n\n', numpic);

%% 3. Initialization and Memory Cache
fprintf('--- 3. Initializing Dictionary, Sparse Codes, and Memory Cache ---\n');
d = patch_size^2;
D_L = randn(d, m);
D_L = proxomiga2_2D(D_L); % L2 norm projection
alpha_cells = cell(1, numpic);
for l = 1:numpic
    alpha_cells{l} = zeros(m, N_l_vec(l)); 
end

% Initialize historical step size memory (Warm Start)
tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

% Log allocation
history.total_objective = zeros(1, MAXITER);
history.data_error = zeros(1, MAXITER);
history.l1_penalty = zeros(1, MAXITER);
history.sparsity = zeros(1, MAXITER);
history.time = zeros(1, MAXITER);

%% 4. V-BASTA Main Optimization Loop
fprintf('--- 4. Starting V-BASTA Optimization Loop ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: Update Sparse Coefficients (alpha) via Variable Metric S_alpha
    % ==========================================================
    abs_D = abs(D_L);
    D_sum = sum(abs_D, 2); 
    Dt = D_L'; 
    
    % --- Core Optimization: Intelligent Caching of S_alpha ---
    S_alpha_cached_list = cell(1, numpic);
    unique_sizes = [];
    unique_S_alpha = {};
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
        y_l = y_cells{l}; 
        img_size_l = image_sizes{l}; 
        
        S_alpha_l = S_alpha_cached_list{l};
        [grad_alpha_l, ~, H_old] = compute_patch_model_gradients(D_L, Dt, old_alpha_l, y_l, img_size_l, patch_size);
        
        tau1_curr = max(gamma1, tau1_history(l) / 1.2); 
        alpha_l_candidate = old_alpha_l; 
        
        for k_bt = 1:max_backtrack_iter
            S_effective = tau1_curr * S_alpha_l; 
            
            v_alpha = old_alpha_l - grad_alpha_l ./ S_effective;
            lambda_th = lambda1 ./ S_effective; 
            alpha_l_candidate = soft_threshold(v_alpha, lambda_th);
            
            y_recon_new = mycol2im(D_L * alpha_l_candidate, img_size_l, patch_size);
            f_val = 0.5 * sum((y_recon_new - y_l).^2, 'all');
            Q_val = H_old + compute_descent_lemma_upper_bound_VM(old_alpha_l, alpha_l_candidate, grad_alpha_l, S_effective);
            
            if f_val <= Q_val
                tau1_history(l) = tau1_curr; 
                break; 
            else
                tau1_curr = tau1_curr * tau1; 
            end
        end
        
        % Failure handling for Reviewer #2 reproducibility request
        if k_bt == max_backtrack_iter
            warning('Maximum backtracking steps reached for sample %d. Descent condition not strictly met.', l);
        end
        
        new_alpha_cells{l} = alpha_l_candidate;
    end
    alpha_cells = new_alpha_cells; 
    
    % ==========================================================
    % Phase 2: Update Dictionary (D_L) via Variable Metric S_D
    % ==========================================================
    [~, grad_D_H_total, H_old_D_update] = compute_H_gradients_in_parallel(D_L, alpha_cells, y_cells, image_sizes, patch_size);
    
    % --- Compute Variable Metric Matrix S_D ---
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
            H_new_D_update = H_new_D_update + 0.5 * sum((y_recon_l - y_cells{l}).^2, 'all'); 
        end
        f_val = H_new_D_update; 
        Q_val = H_old_D_update + compute_descent_lemma_upper_bound_VM(old_D_L, D_candidate, grad_D_H_total, S_effective_D);
        
        if f_val <= Q_val
            tau2_history = tau2_curr; 
            break; 
        else
            tau2_curr = tau2_curr * tau2; 
        end
    end
    
    % Failure handling
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
    
    fprintf('Iter %04d: Obj=%.3e (Data:%.3e, L1:%.3e), Sparsity=%.3f%%, Time=%.2fs\n', ...
            outerIter, total_objective, data_error_term, l1_penalty_term, history.sparsity(outerIter)*100, history.time(outerIter));
end

%% 5. Save Results
SAVE_FILE = 'vbasta_l1_results.mat';
fprintf('\n--- 5. Saving V-BASTA Pre-training Results to %s ---\n', SAVE_FILE);
save(SAVE_FILE, 'D_L', 'alpha_cells', 'history');
fprintf('Results saved successfully.\n\n');

%% 6. Final Evaluation and Visualization
fprintf('--- 6. Computing Final Evaluation Metrics and Visualization ---\n');
sum_psnr = 0;
sum_ssim = 0;
for i = 1:numpic
    y_recon_i = mycol2im(D_L * alpha_cells{i}, image_sizes{i}, patch_size);
    sum_psnr = sum_psnr + psnr(y_cells{i}, y_recon_i);
    sum_ssim = sum_ssim + ssim(y_recon_i, y_cells{i});
end
avg_psnr = sum_psnr / numpic;
avg_ssim = sum_ssim / numpic;

fprintf('======================================================\n');
fprintf('Final Results Summary (%d Iterations):\n', MAXITER);
fprintf('Total Objective (Obj):       %.3e\n', history.total_objective(end));
fprintf('Data Fidelity (Data):        %.3e\n', history.data_error(end));
fprintf('Regularization (L1):         %.3e\n', history.l1_penalty(end));
fprintf('Sparsity:                    %.3f%%\n', history.sparsity(end)*100);
fprintf('Total Time (s):              %.3f\n', history.time(end));
fprintf('Average-PSNR (dB):           %.2f\n', avg_psnr);
fprintf('Average-SSIM :               %.2f\n', avg_ssim);
fprintf('======================================================\n\n');

% --- Figure 1: Detailed Convergence Curves ---
figure('Name', 'V-BASTA - Detailed Convergence Curves');
plot(1:MAXITER, history.total_objective, 'b-', 'LineWidth', 2);
hold on;
plot(1:MAXITER, history.data_error, 'r--');
plot(1:MAXITER, history.l1_penalty, 'g:');
hold off;
xlabel('Iterations'); ylabel('Objective Value'); title('V-BASTA Detailed Convergence Curves');
legend('Total Objective', 'Data Fidelity', 'L1 Penalty');
grid on; set(gca, 'YScale', 'log');

% --- Figure 2: Sparsity Evolution ---
figure('Name', 'V-BASTA - Sparsity Evolution');
plot(1:MAXITER, history.sparsity * 100, 'm-', 'LineWidth', 1.5);
xlabel('Iterations'); ylabel('Sparsity (%)'); title('V-BASTA Sparsity Evolution Curve');
grid on;

% --- Figure 3: Image Reconstruction and Residual ---
img_idx_to_show = 1;
y_original = y_cells{img_idx_to_show};
alpha_final = alpha_cells{img_idx_to_show};
img_size = image_sizes{img_idx_to_show};
y_reconstructed = mycol2im(D_L * alpha_final, img_size, patch_size);
residual_image = abs(y_original - y_reconstructed);
psnr_val = psnr(y_original, y_reconstructed);

figure('Name', 'V-BASTA - Image Reconstruction Details');
subplot(1, 3, 1); imshow(y_original, []); title('Original Image');
subplot(1, 3, 2); imshow(y_reconstructed, []); title(sprintf('Reconstructed Image (PSNR: %.2f dB)', psnr_val));
subplot(1, 3, 3); imshow(residual_image, []); title('Absolute Residual Image'); colorbar;

% --- Figure 4: Learned Dictionary ---
figure('Name', 'V-BASTA - Learned Dictionary');
if exist('showDictionary', 'file')
    showDictionary(D_L);
    title(sprintf('V-BASTA Learned Dictionary (m=%d, patch\\_size=%d)', m, patch_size));
else
    warning('The function showDictionary was not found. Skipping dictionary visualization.');
end

% --- Figure 5: Sparse Code Map ---
figure('Name', 'V-BASTA - Sparse Code Map');
imagesc(abs(alpha_final));
xlabel('Patch Index'); ylabel('Atom Index');
title('Absolute Sparse Coefficients of the First Image');
colorbar;