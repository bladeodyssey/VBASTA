%% Variable-metric Backtracking Alternating Spatially-aware Thresholding Algorithm (V-BASTA) for L1 Image Denoising
% =========================================================================
% Features:
% 1. Performs L1 denoising under the spatially adaptive variable-metric framework.
% 2. Real-time monitoring of both Noisy-PSNR and True-PSNR to prevent overfitting to noise.
% 3. Saves the denoising environment to allow seamless hot-starting for L0 fine-tuning.
%revised 2026.6.17 Minghui Xu
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. Parameter Settings
fprintf('--- 1. Parameter Settings (V-BASTA L1 Denoising Pre-training) ---\n');
m = 100;              % Number of dictionary atoms
patch_size = 11;      % Spatial filter size
sigma_noise = 25;     % Simulated noise standard deviation (e.g., 25/255)

% [Crucial Parameter] L1 Regularization Parameter for Denoising
% Unlike pure dictionary learning, lambda1 must be tuned carefully to avoid 
% over-smoothing the image and losing fine structural details.
lambda1 = 0.2;        

% Denoising is prone to noise overfitting, so MAXITER can be relatively small for early stopping
MAXITER = 100;       
max_backtrack_iter = 50; 
gamma1 = 0.006;       % Initial metric scaling parameter for S_alpha
gamma2 = 0.006;       % Initial metric scaling parameter for S_D
tau1 = 1.5;           % Backtracking decay multiplier for alpha update
tau2 = 1.5;           % Backtracking decay multiplier for D_L update

%% 2. Load Images and Add Noise
fprintf('--- 2. Loading Clean Data and Adding Gaussian Noise ---\n');
try 
    L_data = CreateImages('datasets/denoisy', 'none', 0, 'gray'); 
    y_clean_cells = squeeze(num2cell(L_data, [1, 2, 3])); 
catch
    warning('Loading failed. Using randomly generated test data instead.'); 
    for i=1:2, y_clean_cells{i} = rand(100, 100); end 
end
numpic = length(y_clean_cells);

y_noisy_cells = cell(1, numpic);
noisy_psnr_initial = zeros(1, numpic);
image_sizes = cell(1, numpic); 
N_l_vec = zeros(1, numpic); 

for l = 1:numpic
    y_clean = y_clean_cells{l};
    % Add Additive White Gaussian Noise (AWGN)
    noise = (sigma_noise / 255) * randn(size(y_clean));
    y_noisy = y_clean + noise;
    y_noisy_cells{l} = y_noisy;
    
    % Calculate initial PSNR of the noisy image
    noisy_psnr_initial(l) = psnr(y_clean, y_noisy);
    
    % Precompute dimensions
    image_sizes{l} = [size(y_noisy, 1), size(y_noisy, 2)]; 
    dummy_patches = myim2col(zeros(image_sizes{l}), patch_size); 
    N_l_vec(l) = size(dummy_patches, 2); 
end
fprintf('Initial Average PSNR of Noisy Images: %.2f dB\n\n', mean(noisy_psnr_initial));

%% 3. Initialization and Memory Cache
fprintf('--- 3. Initializing Dictionary, Sparse Codes, and Memory Cache ---\n');
d = patch_size^2;
D_L = randn(d, m);
if exist('proxomiga2_2D', 'file')
    D_L = proxomiga2_2D(D_L); 
else
    D_L = D_L ./ max(1e-12, vecnorm(D_L, 2, 1));
end

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
history.avg_psnr = zeros(1, MAXITER);  % Fit to the noisy observation
history.true_psnr = zeros(1, MAXITER); % Actual denoising capability against ground truth

%% 4. V-BASTA Denoising Main Loop
fprintf('--- 4. Starting V-BASTA L1 Denoising Iteration Loop ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: Update Sparse Coefficients (alpha) fitting y_noisy_cells
    % ==========================================================
    abs_D = abs(D_L);
    D_sum = sum(abs_D, 2); 
    Dt = D_L'; 
    
    % --- Intelligent Caching for S_alpha ---
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
        y_noisy_l = y_noisy_cells{l}; 
        img_size_l = image_sizes{l}; 
        
        S_alpha_l = S_alpha_cached_list{l};
        [grad_alpha_l, ~, H_old] = compute_patch_model_gradients(D_L, Dt, old_alpha_l, y_noisy_l, img_size_l, patch_size);
        
        tau1_curr = max(gamma1, tau1_history(l) / 1.2); 
        alpha_l_candidate = old_alpha_l; 
        
        for k_bt = 1:max_backtrack_iter
            S_effective = tau1_curr * S_alpha_l; 
            v_alpha = old_alpha_l - grad_alpha_l ./ S_effective;
            
            % L1 Soft Thresholding
            lambda_th = lambda1 ./ S_effective; 
            alpha_l_candidate = soft_threshold(v_alpha, lambda_th);
            
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
    l1_penalty_term = lambda1 * sum(abs(cell2mat(alpha_cells)), 'all');
    total_objective = data_error_term + l1_penalty_term;

    history.total_objective(outerIter) = total_objective;
    history.data_error(outerIter) = data_error_term;
    history.l1_penalty(outerIter) = l1_penalty_term;
    history.sparsity(outerIter) = sum(cellfun(@nnz, alpha_cells)) / (sum(N_l_vec)*m);
    
    % Calculate current Noisy-PSNR and True-PSNR
    sum_noisy_psnr = 0;
    sum_true_psnr = 0;
    for l = 1:numpic
        y_recon_l = mycol2im(D_L * alpha_cells{l}, image_sizes{l}, patch_size);
        sum_noisy_psnr = sum_noisy_psnr + psnr(y_noisy_cells{l}, y_recon_l);
        sum_true_psnr  = sum_true_psnr  + psnr(y_clean_cells{l}, y_recon_l);
    end
    history.avg_psnr(outerIter) = sum_noisy_psnr / numpic;
    history.true_psnr(outerIter) = sum_true_psnr / numpic;
    
    fprintf('Iter %04d: Obj=%.2e, Noisy-PSNR=%.2f dB, True-PSNR=%.2f dB, Sparsity=%.2f%%, Time=%.2fs\n', ...
            outerIter, total_objective, history.avg_psnr(outerIter), history.true_psnr(outerIter), history.sparsity(outerIter)*100, iter_time);
end

%% 5. Save Results for L0 Fine-Tuning
SAVE_FILE = 'vbasta_l1_denoising_results.mat';
fprintf('\n--- 5. Saving V-BASTA Denoising Pre-training Results to %s ---\n', SAVE_FILE);
save(SAVE_FILE, 'y_clean_cells', 'y_noisy_cells', 'D_L', 'alpha_cells', 'image_sizes', 'N_l_vec', 'history');
fprintf('Results saved successfully.\n\n');

%% 6. Final Visual Evaluation
fprintf('--- 6. Computing Final Evaluation Metrics and Visualization ---\n');
img_idx_to_show = 1;
y_clean_target = y_clean_cells{img_idx_to_show};
y_noisy_input  = y_noisy_cells{img_idx_to_show};
alpha_final = alpha_cells{img_idx_to_show};
img_size = image_sizes{img_idx_to_show};

y_denoised = mycol2im(D_L * alpha_final, img_size, patch_size);
psnr_noisy = psnr(y_clean_target, y_noisy_input);
psnr_denoised = psnr(y_clean_target, y_denoised);
% ssim_denoised = ssim(y_denoised, y_clean_target); % Enable if SSIM is needed

% --- Figure 1: Denoising Comparison ---
figure('Name', 'V-BASTA - Denoising Results', 'Position', [100, 200, 1200, 400]);
subplot(1, 3, 1); imshow(y_clean_target, []); title('Original Clean Image');
subplot(1, 3, 2); imshow(y_noisy_input, []); title(sprintf('Noisy Image\nPSNR: %.2f dB', psnr_noisy));
subplot(1, 3, 3); imshow(y_denoised, []); title(sprintf('V-BASTA L1 Denoised\nPSNR: %.2f dB', psnr_denoised));

% --- Figure 2: PSNR Evolution ---
figure('Name', 'V-BASTA - PSNR Evolution');
plot(1:MAXITER, history.avg_psnr, 'r--', 'LineWidth', 1.5); hold on;
plot(1:MAXITER, history.true_psnr, 'g-', 'LineWidth', 2);
xlabel('Iterations'); ylabel('Average PSNR (dB)'); title('V-BASTA L1 Denoising PSNR Evolution');
legend('Noisy-PSNR (Model fitting to noisy image)', 'True-PSNR (Actual denoising capability)', 'Location', 'best');
grid on;

% --- Figure 3: Learned Dictionary ---
figure('Name', 'V-BASTA - Learned Dictionary');
if exist('showDictionary', 'file')
    showDictionary(D_L);
    title(sprintf('Learned Dictionary from Noisy Images (m=%d)', m));
else
    imagesc(D_L); colormap gray; title('Learned Dictionary Matrix');
end