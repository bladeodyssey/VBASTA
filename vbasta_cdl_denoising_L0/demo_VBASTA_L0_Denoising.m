%% Variable-metric Backtracking Alternating Spatially-aware Thresholding Algorithm (V-BASTA) for L0 Image Denoising
% =========================================================================
% Features:
% 1. Seamless Hot-Start: Loads saved results from the L1 denoising phase.
% 2. Optimization Strategy: Gradients and data fidelity are computed against y_noisy_cells.
% 3. Real-time Monitoring: Tracks True-PSNR to prevent L0 overfitting to noise.
% 4. Adaptive Metric: Accelerates L0 dynamic hard thresholding via Gerschgorin bounds.
%revised 2026.6.17 Minghui Xu
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. Core Parameter Settings
fprintf('--- 1. Parameter Settings (V-BASTA L0 Denoising Fine-tuning) ---\n');
L1_RESULTS_FILE = 'vbasta_l1_denoising_results.mat'; 
% (Note: If using old pre-training results, change to 'vmanalp_l1_denoising_results.mat')

% [Core Denoising Parameter]: lambda0 determines the dynamic hard thresholding intensity.
% Increase lambda0 if residual noise remains; decrease if structural details are lost.
lambda0 = 0.0002;        

m = 100;              
patch_size = 11;      

% L0 fine-tuning can overfit noise easily. MAXITER is kept extremely small for hot-start.
MAXITER = 1;       
max_backtrack_iter = 50; 
gamma1 = 0.0001;         
gamma2 = 0.0001;         
tau1 = 1.5;           
tau2 = 1.5;           

%% 2. Load L1 Hot-Start Data and Preprocess
fprintf('--- 2. Loading L1 Hot-Start Data ---\n');
if ~exist(L1_RESULTS_FILE, 'file')
    error('File %s not found. Please ensure the V-BASTA L1 denoising script has been executed and saved!', L1_RESULTS_FILE);
end

% Load workspace containing clean images, noisy images, pre-trained dictionary, and coefficients
loaded_data = load(L1_RESULTS_FILE);

% Robust compatibility check for variable names
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
y_noisy_cells = loaded_data.y_noisy_cells;
image_sizes = loaded_data.image_sizes;
N_l_vec = loaded_data.N_l_vec;

numpic = length(y_clean_cells);
d = patch_size^2;

fprintf('Hot-start loaded successfully! Seamlessly transitioning from L1 results.\n');

% Calculate the initial True-PSNR from the hot-start phase
initial_true_psnr = zeros(1, numpic);
for l = 1:numpic
    y_l1_recon = mycol2im(D_L * alpha_cells{l}, image_sizes{l}, patch_size);
    initial_true_psnr(l) = psnr(y_clean_cells{l}, y_l1_recon);
end
fprintf('Average True-PSNR at the end of the L1 phase: %.4f dB\n\n', mean(initial_true_psnr));

%% 3. Initialization and Memory Cache
fprintf('--- 3. Initializing Variables and Memory Cache ---\n');
% Initialize historical step size memory
tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

history.total_objective = zeros(1, MAXITER);
history.data_error = zeros(1, MAXITER);
history.l0_penalty = zeros(1, MAXITER);
history.sparsity = zeros(1, MAXITER);
history.time = zeros(1, MAXITER); 
history.avg_psnr = zeros(1, MAXITER);  % Noisy-PSNR
history.true_psnr = zeros(1, MAXITER); % True-PSNR

%% 4. V-BASTA Denoising Main Loop
fprintf('--- 4. Starting V-BASTA L0 Denoising Fine-tuning (Dynamic Hard Thresholding) ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: Update Sparse Coefficients (alpha) fitting noisy images
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
    
    new_alpha_cells = cell(1, numpic);
    
    parfor l = 1:numpic
        old_alpha_l = alpha_cells{l}; 
        y_noisy_l = y_noisy_cells{l}; % [Core]: Target is the noisy image
        img_size_l = image_sizes{l}; 
        
        S_alpha_l = S_alpha_cached_list{l};
        [grad_alpha_l, ~, H_old] = compute_patch_model_gradients(D_L, Dt, old_alpha_l, y_noisy_l, img_size_l, patch_size);
        
        tau1_curr = max(gamma1, tau1_history(l) / 1.2); 
        alpha_l_candidate = old_alpha_l; 
        
        for k_bt = 1:max_backtrack_iter
            S_effective = tau1_curr * S_alpha_l; 
            
            v_alpha = old_alpha_l - grad_alpha_l ./ S_effective;
            
            % Dynamic Hard Thresholding under Variable Metric
            lambda_th = sqrt(2 * lambda0 ./ S_effective); 
            alpha_l_candidate = v_alpha .* (abs(v_alpha) > lambda_th);
            
            y_recon_new = mycol2im(D_L * alpha_l_candidate, img_size_l, patch_size);
            f_val = 0.5 * sum((y_recon_new - y_noisy_l).^2, 'all');
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
    % Phase 2: Update Dictionary (D_L)
    % ==========================================================
    [~, grad_D_H_total, H_old_D_update] = compute_H_gradients_in_parallel(D_L, alpha_cells, y_noisy_cells, image_sizes, patch_size);
    
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
        if exist('prox_vmetric_L2_ball', 'file')
            D_candidate = prox_vmetric_L2_ball(v_D, S_effective_D);
        else
            D_candidate = proxomiga2_2D(v_D);
        end
        
        H_new_D_update = 0;
        for l = 1:numpic
            y_recon_l = mycol2im(D_candidate * alpha_cells{l}, image_sizes{l}, patch_size); 
            H_new_D_update = H_new_D_update + 0.5 * sum((y_recon_l - y_noisy_cells{l}).^2, 'all'); 
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
    % Phase 3: Logging and Denoising Evaluation
    % ==========================================================
    iter_time = toc;
    if outerIter == 1
        history.time(outerIter) = iter_time;
    else
        history.time(outerIter) = history.time(outerIter-1) + iter_time;
    end
    
    data_error_term = H_new_D_update; 
    non_zero_count = sum(cellfun(@nnz, alpha_cells));
    l0_penalty_term = lambda0 * non_zero_count; 
    total_objective = data_error_term + l0_penalty_term;

    history.total_objective(outerIter) = total_objective;
    history.data_error(outerIter) = data_error_term;
    history.l0_penalty(outerIter) = l0_penalty_term;
    history.sparsity(outerIter) = non_zero_count / (sum(N_l_vec)*m);
    
    % Calculate True-PSNR
    sum_noisy_psnr = 0;
    sum_true_psnr = 0;
    for l = 1:numpic
        y_recon_l = mycol2im(D_L * alpha_cells{l}, image_sizes{l}, patch_size);
        sum_noisy_psnr = sum_noisy_psnr + psnr(y_noisy_cells{l}, y_recon_l);
        sum_true_psnr  = sum_true_psnr  + psnr(y_clean_cells{l}, y_recon_l);
    end
    history.avg_psnr(outerIter) = sum_noisy_psnr / numpic;
    history.true_psnr(outerIter) = sum_true_psnr / numpic;
    
    fprintf('Iter %04d: Obj=%.3e, Noisy-PSNR=%.2f, True-PSNR=%.2f, Sparsity=%.3f%%, L0_Count=%d, Time=%.2fs\n', ...
            outerIter, total_objective, history.avg_psnr(outerIter), history.true_psnr(outerIter), history.sparsity(outerIter)*100, non_zero_count, iter_time);
end

%% 5. Final Visual Evaluation
fprintf('\n--- 5. Final Denoising Evaluation and Visualization ---\n');
img_idx_to_show = 1;
y_clean_target = y_clean_cells{img_idx_to_show};
y_noisy_input  = y_noisy_cells{img_idx_to_show};

% Load and reconstruct from the L1 hot-start file solely for comparison
loaded_l1_data = load(L1_RESULTS_FILE); 
if isfield(loaded_l1_data, 'D_L'), D_L_l1 = loaded_l1_data.D_L; else, D_L_l1 = loaded_l1_data.D; end
if isfield(loaded_l1_data, 'alpha_cells'), alpha_cells_l1 = loaded_l1_data.alpha_cells; else, alpha_cells_l1 = loaded_l1_data.A_cells; end

y_l1_recon = mycol2im(D_L_l1 * alpha_cells_l1{img_idx_to_show}, image_sizes{img_idx_to_show}, patch_size);
y_l0_recon = mycol2im(D_L * alpha_cells{img_idx_to_show}, image_sizes{img_idx_to_show}, patch_size);

psnr_noisy = psnr(y_clean_target, y_noisy_input);
psnr_l1    = psnr(y_clean_target, y_l1_recon);
psnr_l0    = psnr(y_clean_target, y_l0_recon);

% --- Figure 1: Four-Panel Denoising Comparison ---
figure('Name', 'V-BASTA L0 - Denoising Comparison', 'Position', [100, 200, 1500, 400]);
subplot(1, 4, 1); imshow(y_clean_target, []); title('Original Clean Image');
subplot(1, 4, 2); imshow(y_noisy_input, []); title(sprintf('Noisy Image\nPSNR: %.2f dB', psnr_noisy));
subplot(1, 4, 3); imshow(y_l1_recon, []); title(sprintf('L1 Coarse Denoising (Hot-Start)\nPSNR: %.2f dB', psnr_l1));
subplot(1, 4, 4); imshow(y_l0_recon, []); title(sprintf('V-BASTA L0 Fine-tuning\nPSNR: %.2f dB', psnr_l0));

% --- Figure 2: PSNR Evolution Curve ---
figure('Name', 'V-BASTA L0 - PSNR Evolution');
plot(1:MAXITER, history.avg_psnr, 'r--', 'LineWidth', 1.5); hold on;
plot(1:MAXITER, history.true_psnr, 'g-', 'LineWidth', 2);
yline(mean(initial_true_psnr), 'b:', 'Final PSNR from L1 Phase', 'LineWidth', 1.5);
xlabel('L0 Iteration'); ylabel('Average PSNR (dB)'); title('V-BASTA L0 Denoising PSNR Evolution');
legend('Noisy-PSNR (Model fitting to noisy image)', 'True-PSNR (Actual denoising capability)', 'Location', 'best');
grid on;

% --- Figure 3: Sparsity and Objective Evolution ---
figure('Name', 'V-BASTA L0 - Sparsity Evolution');
yyaxis left;
plot(1:MAXITER, history.sparsity * 100, 'm-', 'LineWidth', 1.5);
ylabel('Sparsity (%)');
yyaxis right;
plot(1:MAXITER, history.total_objective, 'b-', 'LineWidth', 1.5);
ylabel('Objective Value');
xlabel('Iterations'); title('Sparsity and Objective Function Evolution');
grid on;