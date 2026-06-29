%% Variable-metric Backtracking Alternating Spatially-aware Thresholding Algorithm (V-BASTA) for L0-CDL
% =========================================================================
% Features:
% 1. Hot-Start: Smooth transition from L1 continuous relaxation to strict L0 sparsity.
% 2. Adaptive Metric: Accelerated descent using Gerschgorin variable-metric matrices.
% 3. Dynamic Hard Thresholding: Strict L0 proximal operator under variable metric 
%    for spatially-aware support truncation.
% 4. Dimension-safe caching mechanism for S_alpha.
% 5. Dual Objective Logging: Records both the optimization-driven objective 
%    and the standard reporting objective for paper evaluation.
%revised 2026.6.17 Minghui Xu
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. Parameter Settings
fprintf('--- 1. Parameter Settings (V-BASTA L0 Phase - Dynamic Hard Thresholding) ---\n');
USE_HOT_START = true; 
% Path to the pre-trained L1 results file
L1_RESULTS_FILE = 'D:\desk\论文\MATLABcdl env\VMANALP_EX\ablation\robust\L1\change gamma\l11001105gamma=0.1.mat'; 

m = 100;              % Number of dictionary atoms
patch_size = 11;      % Spatial filter size

% [Crucial Parameter] Internal optimization-driven parameter lambda0.
% Adjust this to control the actual truncation threshold and desired sparsity level (e.g., 0.066%).
lambda0 = 0.5;   

% V-BASTA specific backtracking and metric scaling parameters
MAXITER = 5;       
max_backtrack_iter = 50; 
gamma1 = 0.1;         
gamma2 = 0.1;         
tau1 = 1.5;           
tau2 = 1.5;           

%% 2. Data Loading and Preprocessing
fprintf('--- 2. Loading and Preprocessing Data ---\n');
try 
    L_data = CreateImages('datasets/city_100_100','local_cn',1,'gray'); 
    y_cells = squeeze(num2cell(L_data,[1,2,3])); 
catch
    warning('Loading failed. Using randomly generated test data instead.'); 
    for i=1:2, y_cells{i} = rand(100,100); end 
end
numpic = length(y_cells);

image_sizes = cell(1, numpic); 
N_l_vec = zeros(1, numpic); 
for l = 1:numpic
    y_l = y_cells{l}; 
    image_sizes{l} = [size(y_l,1), size(y_l,2)]; 
    dummy_patches = myim2col(zeros(image_sizes{l}), patch_size); 
    N_l_vec(l) = size(dummy_patches, 2); 
end
fprintf('Preprocessing complete. Loaded %d images.\n\n', numpic);

%% 3. Initialization (Robust Hot-Start Logic)
fprintf('--- 3. Variable Initialization (Hot-Start Detection) ---\n');
d = patch_size^2;

if USE_HOT_START && exist(L1_RESULTS_FILE, 'file')
    fprintf('Hot-start option detected. Loading and verifying from %s...\n', L1_RESULTS_FILE);
    loaded_data = load(L1_RESULTS_FILE); 
    is_compatible = true;
    
    % Compatability check: Handles both old variable names (D, A_cells) and new ones (D_L, alpha_cells)
    if isfield(loaded_data, 'D_L')
        D_L_loaded = loaded_data.D_L;
    elseif isfield(loaded_data, 'D')
        D_L_loaded = loaded_data.D;
    else
        warning('Dictionary variable not found in the loaded file.'); is_compatible = false;
    end
    
    if isfield(loaded_data, 'alpha_cells')
        alpha_cells_loaded = loaded_data.alpha_cells;
    elseif isfield(loaded_data, 'A_cells')
        alpha_cells_loaded = loaded_data.A_cells;
    else
        warning('Sparse coefficients variable not found in the loaded file.'); is_compatible = false;
    end
    
    % Dimension verification
    if is_compatible && ~(ismatrix(D_L_loaded) && size(D_L_loaded,1)==d && size(D_L_loaded,2)==m)
        warning('Dimension mismatch for the loaded dictionary D_L!'); is_compatible = false; 
    end
    if is_compatible && ~(iscell(alpha_cells_loaded) && length(alpha_cells_loaded)==numpic)
        warning('Image count mismatch for the loaded sparse coefficients!'); is_compatible = false; 
    end
    if is_compatible
        for l=1:numpic
            if size(alpha_cells_loaded{l},2) ~= N_l_vec(l)
                warning('Patch count mismatch for sparse coefficients of image %d!', l); is_compatible = false; break; 
            end
        end
    end
    
    if is_compatible
        D_L = D_L_loaded; 
        alpha_cells = alpha_cells_loaded; 
        fprintf('Hot-start loaded successfully! Seamlessly transitioning from L1 results.\n\n'); 
    else
        USE_HOT_START = false; warning('Hot-start verification failed. Falling back to random initialization.'); 
    end
else
    if USE_HOT_START, warning('Hot-start file not found. Falling back to random initialization.'); end
    USE_HOT_START = false;
end

if ~USE_HOT_START
    fprintf('Applying random initialization...\n\n'); 
    D_L = randn(d, m); 
    D_L = prox_vmetric_L2_ball(D_L, ones(d, m)); % Strict spherical projection
    alpha_cells = cell(1, numpic); 
    for l=1:numpic, alpha_cells{l} = zeros(m, N_l_vec(l)); end
end

% Initialize historical step size memory
tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

% History structure for dual objective recording
history.total_objective = zeros(1, MAXITER);   % Optimization-driven Obj (Data + lambda*L0)
history.target_objective = zeros(1, MAXITER);  % Table-reporting Obj (Data + exact L0 count)
history.data_error = zeros(1, MAXITER);
history.l0_penalty = zeros(1, MAXITER);
history.sparsity = zeros(1, MAXITER);
history.time = zeros(1, MAXITER); 

%% 4. V-BASTA Main Optimization Loop
fprintf('--- 4. Starting V-BASTA L0 Optimization Loop (Dynamic Hard Thresholding) ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: Update Sparse Coefficients (alpha) via VM Gradient Descent & L0 Hard Thresholding
    % ==========================================================
    abs_D = abs(D_L);
    D_sum = sum(abs_D, 2); 
    Dt = D_L'; 
    
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
    % ---------------------------------
    
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
            
            % 1. Variable-Metric Gradient Step
            v_alpha = old_alpha_l - grad_alpha_l ./ S_effective;
            
            % 2. Dynamic Hard Thresholding (L0 Proximal Mapping under Variable Metric)
            lambda_th = sqrt(2 * lambda0 ./ S_effective); 
            alpha_l_candidate = v_alpha .* (abs(v_alpha) > lambda_th);
            
            % 3. Evaluate Descent Lemma Quadratic Majorizer
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
        
        % Failure handling
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
        % Exact Newton's method projection for variable-metric L2 ball
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
    % Phase 3: Logging and Dual Objective Evaluation
    % ==========================================================
    iter_time = toc;
    if outerIter == 1
        history.time(outerIter) = iter_time;
    else
        history.time(outerIter) = history.time(outerIter-1) + iter_time;
    end
    
    data_error_term = H_new_D_update; 
    non_zero_count = sum(cellfun(@nnz, alpha_cells)); % Total non-zero elements
    
    % 1. Internal Optimization Objective (Data + lambda0 * ||alpha||_0)
    l0_penalty_term = lambda0 * non_zero_count; 
    total_objective = data_error_term + l0_penalty_term;

    % 2. Table Reporting Objective (Data + 1 * ||alpha||_0)
    target_objective = data_error_term + non_zero_count;

    history.total_objective(outerIter) = total_objective;
    history.target_objective(outerIter) = target_objective; 
    history.data_error(outerIter) = data_error_term;
    history.l0_penalty(outerIter) = l0_penalty_term;
    history.sparsity(outerIter) = non_zero_count / (sum(N_l_vec)*m);
    
    fprintf('Iter %04d: Obj(Table)=%.3e | Obj(Opt)=%.3e | Data:%.3e | L0_Count:%d | Sparsity=%.3f%% | Time=%.2fs\n', ...
            outerIter, target_objective, total_objective, data_error_term, non_zero_count, history.sparsity(outerIter)*100, history.time(outerIter));
end

%% 5. Save Results
SAVE_FILE = 'vbasta_l0_results.mat';
fprintf('\n--- 5. Saving V-BASTA L0 Results to %s ---\n', SAVE_FILE);
save(SAVE_FILE, 'D_L', 'alpha_cells', 'history');
fprintf('Results saved successfully.\n\n');

%% 6. Final Evaluation and Visualization
fprintf('--- 6. Computing Final Evaluation Metrics and Visualization ---\n');
sum_psnr = 0;
sum_ssim = 0;
for i = 1:numpic
    y_original = y_cells{i};
    y_recon_i = mycol2im(D_L * alpha_cells{i}, image_sizes{i}, patch_size);
    
    sum_psnr = sum_psnr + psnr(y_original, y_recon_i);
    % sum_ssim = sum_ssim + ssim(y_recon_i, y_original); % Enable if SSIM is needed
end
avg_psnr = sum_psnr / numpic;
% avg_ssim = sum_ssim / numpic;

fprintf('======================================================\n');
fprintf('Final L0 Results Summary (%d Iterations):\n', MAXITER);
fprintf('Obj for Table (Data + ||alpha||_0): %.3e  <-- Target value for paper table\n', history.target_objective(end));
fprintf('Internal Opt Obj (Data + lam*L0):   %.3e  <-- Internal convergence value\n', history.total_objective(end));
fprintf('------------------------------------------------------\n');
fprintf('Data Fidelity (Data Error):         %.3e\n', history.data_error(end));
fprintf('Non-Zero Elements Count:            %d\n', sum(cellfun(@nnz, alpha_cells)));
fprintf('Sparsity:                           %.3f%%\n', history.sparsity(end)*100);
fprintf('Total Time (s):                     %.3f\n', history.time(end));
fprintf('Average-PSNR (dB):                  %.2f\n', avg_psnr);
fprintf('======================================================\n\n');

% =========================================================================
% Plotting Section: Dual Objective Convergence Visualization
% =========================================================================
figure('Name', 'V-BASTA L0 - Dual Objective Convergence');

% Plot the standard reporting objective (Data + ||alpha||_0)
plot(1:MAXITER, history.target_objective, 'b-', 'LineWidth', 2.5);
hold on;
% Plot the actual objective driving the algorithm (Data + \lambda * ||alpha||_0)
plot(1:MAXITER, history.total_objective, 'm-', 'LineWidth', 1.5);
% Plot data fidelity error solely
plot(1:MAXITER, history.data_error, 'r--', 'LineWidth', 1.5);

hold off;
xlabel('Iterations', 'FontSize', 11, 'FontWeight', 'bold'); 
ylabel('Objective Value (Log Scale)', 'FontSize', 11, 'FontWeight', 'bold'); 
title('V-BASTA (L_0) Objective Function Convergence Comparison', 'FontSize', 12);
legend('Obj for Table (Data Error + ||\alpha||_0)', ...
       'Internal Obj (Data Error + \lambda ||\alpha||_0)', ...
       'Data Error', 'Location', 'northeast', 'FontSize', 10);
grid on; 
set(gca, 'YScale', 'log', 'FontSize', 10);

% --- Figure 2: Sparsity Evolution ---
figure('Name', 'V-BASTA L0 - Sparsity Evolution');
plot(1:MAXITER, history.sparsity * 100, 'm-', 'LineWidth', 2);
xlabel('Iterations'); ylabel('Sparsity (%)'); title('V-BASTA L0 Sparsity Evolution');
grid on;

% --- Figure 3: Image Reconstruction and Residual ---
img_idx_to_show = 1;
y_original = y_cells{img_idx_to_show};
alpha_final = alpha_cells{img_idx_to_show};
img_size = image_sizes{img_idx_to_show};
y_reconstructed = mycol2im(D_L * alpha_final, img_size, patch_size);
residual_image = abs(y_original - y_reconstructed);

figure('Name', 'V-BASTA L0 - Image Reconstruction Details');
subplot(1, 3, 1); imshow(y_original, []); title('Original Image');
subplot(1, 3, 2); imshow(y_reconstructed, []); title(sprintf('Reconstructed Image (PSNR: %.2f dB)', psnr(y_original, y_reconstructed)));
subplot(1, 3, 3); imshow(residual_image, []); title('Absolute Residual Image'); colorbar;

% --- Figure 4: Learned Dictionary ---
if exist('showDictionary', 'file')
    figure('Name', 'V-BASTA L0 - Learned Dictionary');
    showDictionary(D_L);
    title(sprintf('V-BASTA L0 Learned Dictionary (m=%d, patch\\_size=%d)', m, patch_size));
end