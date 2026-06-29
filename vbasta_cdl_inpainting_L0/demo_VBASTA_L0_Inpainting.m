%% Variable-metric Backtracking Alternating Spatially-aware Thresholding Algorithm (V-BASTA) for L0 Image Inpainting
% =========================================================================
% Features:
% 1. Strict Hot-Start: Directly inherits the exact Mask, degraded images, 
%    and dictionary from the L1 phase to ensure an unbiased comparison environment.
% 2. Variable-Metric L0 Proximal Operator: Integrates Gerschgorin bounds to 
%    perform spatially-aware dynamic hard thresholding.
% 3. Dual-Objective Logging: Outputs a rigorous Target Objective (Data Error + ||alpha||_0) 
%    aligned with standard paper benchmarking protocols.
%revised 2026.6.17 Minghui Xu
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. Parameter Settings and Environment Loading
fprintf('--- 1. Environment Setup (V-BASTA L0 Inpainting Phase) ---\n');

% Specify the L1 pre-training results file
L1_RESULTS_FILE = 'vbasta_l1_inpainting_results.mat'; 
% (Note: If you used the old script, change this to 'vmanalp_l1_inpainting_results.mat')

if ~exist(L1_RESULTS_FILE, 'file')
    error('L1 pre-training file %s not found. Please run the L1 script first to generate consistent masks and degraded data!', L1_RESULTS_FILE);
end

fprintf('Loading L1 pre-training data and initiating strict hot-start...\n');
% Robust loading: Compatible with both old (D, A_cells) and new (D_L, alpha_cells) naming conventions
loaded_data = load(L1_RESULTS_FILE);

if isfield(loaded_data, 'D_L')
    D_L = loaded_data.D_L;
else
    D_L = loaded_data.D; 
end

if isfield(loaded_data, 'alpha_cells')
    alpha_cells = loaded_data.alpha_cells;
else
    alpha_cells = loaded_data.A_cells; 
end

y_clean_cells = loaded_data.y_clean_cells;
y_masked_cells = loaded_data.y_masked_cells;
M_cells = loaded_data.M_cells;
image_sizes = loaded_data.image_sizes;
N_l_vec = loaded_data.N_l_vec;

[d, m] = size(D_L);
patch_size = sqrt(d);
numpic = length(y_clean_cells);

fprintf('Environment loaded successfully. Locked the exact Mask (50%% missing) and degraded image states!\n\n');

%% 2. L0 Specific Optimization Parameters
fprintf('--- 2. L0 Dynamic Hard Thresholding Initialization ---\n');

% [Crucial Parameter] Internal optimization-driven parameter lambda0.
% Controls the intensity of the hard thresholding truncation under the variable metric.
lambda0 = 0.00001;        

% Optimization step sizes and backtracking parameters
MAXITER = 1;             % Fast convergence expected under hot-start
max_backtrack_iter = 50; 
gamma1 = 0.0001;         
gamma2 = 0.0001;         
tau1 = 1.5;           
tau2 = 1.5;           

tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

% Allocate dedicated logging arrays for the L0 phase
history_L0.total_objective = zeros(1, MAXITER);
history_L0.data_error = zeros(1, MAXITER);
history_L0.l0_penalty = zeros(1, MAXITER);
history_L0.sparsity = zeros(1, MAXITER);
history_L0.time = zeros(1, MAXITER); 
history_L0.target_objective = zeros(1, MAXITER); 
history_L0.psnr = zeros(1, MAXITER);

%% 3. V-BASTA Inpainting Main Loop (L0 Dynamic Hard Thresholding)
fprintf('--- 3. Starting V-BASTA L0 Inpainting Optimization Loop ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: Update Sparse Coefficients (alpha) via VM + L0 Hard Thresholding + Mask
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
        
        % 1. Compute gradients incorporating the mask
        [grad_alpha_l, ~, H_old] = compute_patch_model_gradients_mask(D_L, Dt, old_alpha_l, y_masked_l, M_l, img_size_l, patch_size);
        
        tau1_curr = max(gamma1, tau1_history(l) / 1.2); 
        alpha_l_candidate = old_alpha_l; 
        
        for k_bt = 1:max_backtrack_iter
            S_effective = tau1_curr * S_alpha_l; 
            v_alpha = old_alpha_l - grad_alpha_l ./ S_effective;
            
            % 2. Dynamic L0 Hard Thresholding under variable metric
            lambda_th = sqrt(2 * lambda0 ./ S_effective); 
            alpha_l_candidate = v_alpha .* (abs(v_alpha) > lambda_th);
            
            % 3. Check descent lemma quadratic majorizer with mask
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
        
        % Failure handling
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
        D_candidate = prox_vmetric_L2_ball(v_D, S_effective_D); % Exact L2 projection
        
        H_new_D_update = 0;
        for l = 1:numpic
            y_recon_l = mycol2im(D_candidate * alpha_cells{l}, image_sizes{l}, patch_size); 
            % Masked residual computation
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
        history_L0.time(outerIter) = iter_time;
    else
        history_L0.time(outerIter) = history_L0.time(outerIter-1) + iter_time;
    end
    
    data_error_term = H_new_D_update; 
    non_zero_count = sum(cellfun(@nnz, alpha_cells)); 
    
    l0_penalty_term = lambda0 * non_zero_count; 
    total_objective = data_error_term + l0_penalty_term;
    target_objective = data_error_term + non_zero_count;

    history_L0.total_objective(outerIter) = total_objective;
    history_L0.target_objective(outerIter) = target_objective; 
    history_L0.data_error(outerIter) = data_error_term;
    history_L0.l0_penalty(outerIter) = l0_penalty_term;
    history_L0.sparsity(outerIter) = non_zero_count / (sum(N_l_vec)*m);
    
    % Calculate Inpainting PSNR (Observed + Restored)
    current_psnr = 0;
    for l = 1:numpic
        y_recon_curr = mycol2im(D_L * alpha_cells{l}, image_sizes{l}, patch_size);
        y_final_curr = M_cells{l} .* y_clean_cells{l} + (1 - M_cells{l}) .* y_recon_curr;
        current_psnr = current_psnr + psnr(y_clean_cells{l}, y_final_curr);
    end
    history_L0.psnr(outerIter) = current_psnr / numpic;
    
    fprintf('Iter %04d: TargetObj=%.3e [Opt=%.3e], L0_Count:%d, PSNR=%.2fdB, Sparsity=%.3f%%, Time=%.2fs\n', ...
            outerIter, target_objective, total_objective, non_zero_count, history_L0.psnr(outerIter), history_L0.sparsity(outerIter)*100, history_L0.time(outerIter));
end

%% 4. Final Evaluation and Visualization
fprintf('\n--- 4. Final Evaluation Metrics Summary ---\n');

fprintf('======================================================\n');
fprintf('Final L0 Inpainting Results Summary (%d Iterations):\n', MAXITER);
fprintf('Obj for Table (Data + ||alpha||_0): %.3e  <-- Target value for paper table\n', history_L0.target_objective(end));
fprintf('------------------------------------------------------\n');
fprintf('Non-Zero Elements Count:            %d\n', non_zero_count);
fprintf('Sparsity:                           %.3f%%\n', history_L0.sparsity(end)*100);
fprintf('Total Time (s):                     %.3f\n', history_L0.time(end));
fprintf('Average-PSNR (dB):                  %.2f\n', history_L0.psnr(end));
fprintf('======================================================\n\n');

% --- Figure 1: Convergence and Non-Zero Count Evolution ---
figure('Name', 'V-BASTA L0 Inpainting - Convergence');
subplot(2,1,1);
plot(1:MAXITER, history_L0.target_objective, 'b-', 'LineWidth', 2); hold on;
plot(1:MAXITER, history_L0.data_error, 'r--'); 
plot(1:MAXITER, history_L0.target_objective - history_L0.data_error, 'g:'); hold off;
xlabel('Iterations'); ylabel('Objective Value'); title('Objective Function Convergence (Data + L0\_Count)');
legend('Target Obj', 'Data Error', 'L0 Count');
grid on; set(gca, 'YScale', 'log');

subplot(2,1,2);
plot(1:MAXITER, history_L0.psnr, 'm-', 'LineWidth', 2);
xlabel('Iterations'); ylabel('PSNR (dB)'); title('Image Inpainting PSNR Evolution');
grid on;

% --- Figure 2: Image Reconstruction Visual Comparison ---
img_idx_to_show = 1;
y_original = y_clean_cells{img_idx_to_show};
y_masked = y_masked_cells{img_idx_to_show};
M_mask = M_cells{img_idx_to_show};
alpha_final = alpha_cells{img_idx_to_show};
img_size = image_sizes{img_idx_to_show};

y_reconstructed = mycol2im(D_L * alpha_final, img_size, patch_size);
y_final = M_mask .* y_original + (1 - M_mask) .* y_reconstructed;
final_psnr = psnr(y_original, y_final);
final_ssim = ssim(y_final, y_original);

figure('Name', 'V-BASTA L0 Inpainting - Visual Results', 'Position', [100, 100, 1200, 400]);
subplot(1, 4, 1); imshow(y_original, []); title('Original Image');
subplot(1, 4, 2); imshow(y_masked, []); title('Degraded Image (50% Missing)');
subplot(1, 4, 3); imshow(y_reconstructed, []); title('Pure Dictionary L0 Reconstruction');
subplot(1, 4, 4); imshow(y_final, []); title(sprintf('Final Inpainting Result\n(PSNR: %.2f dB)', final_psnr));