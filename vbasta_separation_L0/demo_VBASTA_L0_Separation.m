%% Cartoon-Texture Separation using V-BASTA Framework (L0 Version)
% =========================================================================
% Version: V2.0 (V-BASTA L0 Hot-Start Edition)
% Features:
% 1. L0-Norm Constraint: Replaces L1 soft-thresholding with dynamic hard 
%    thresholding + Top-K truncation for extreme sparsity.
% 2. Warm Start: Initializes D_L and alpha from pre-trained V-BASTA L1 results.
% 3. Best Tracking: Independently tracks the highest PSNR for Cartoon and Texture,
%    supporting early stopping after consecutive iterations without improvement.
% 4. Real-time Monitoring: Four-panel visualization (Separation, Dictionary, 
%    PSNR Evolution, Sparsity Monitoring).
%revise 2026.6.17 Minghui Xu
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. Parameter Settings
fprintf('--- 1. Parameter Settings (V-BASTA L0 Separation) ---\n');
m = 100;              
patch_size = 11;      

% --- L0 Model Parameters ---
lambda0_penalty = 0.001; % L0 sparsity weight (smaller than L1 due to hard thresholding conditions)
sparsity_k = 10;         % Maximum non-zero elements per sparse coding vector (patch)
zeta = 0.015;            % TV smoothness weight for cartoon component

% --- V-BASTA Optimization Parameters ---
MAXITER = 300;           % Fast convergence expected under hot-start; fewer max iterations needed
max_backtrack_iter = 30; 
gamma1 = 0.006;          % Initial scaling for S_alpha
gamma2 = 0.006;          % Initial scaling for S_D
tau_bt = 1.5;            % Backtracking decay multiplier

% --- Visualization & Early Stopping Parameters ---
plot_interval = 5; 
patience = 30;     

%% 2. Load Data
fprintf('--- 2. Loading Data ---\n');
load('datasets/separation/cat4.mat'); 
load('datasets/separation/fense2.mat'); 

I_true = im2double(imgc); 
M_true = 0.4 * im2double(imresize(fense2, [256, 256]));
y = I_true + M_true; 
numpic = 1; 

image_sizes{1} = size(y);
dummy_patches = myim2col(zeros(image_sizes{1}), patch_size);
N_l_vec(1) = size(dummy_patches, 2);

%% 3. Initialization (Hot-Start from L1)
fprintf('--- 3. Initialization (Hot-Start) ---\n');
d = patch_size^2;

try
    fprintf('Loading L1 pre-training optimal results (vbasta_l1_separation_results.mat)...\n');
    loaded_data = load('vbasta_l1_separation_results.mat');
    
    if isfield(loaded_data, 'best_D_L_for_texture')
        D_L = loaded_data.best_D_L_for_texture;
        alpha_cells = loaded_data.best_alpha_for_texture;
    else
        D_L = loaded_data.best_D_for_texture;
        alpha_cells = loaded_data.best_A_for_texture;
    end
    fprintf('Successfully loaded optimal D_L and alpha_cells.\n');
catch
    warning('L1 pre-training results not found. Falling back to random initialization! (Running L1 script first is recommended)');
    D_L = randn(d, m); D_L = proxomiga2_2D(D_L);
    alpha_cells{1} = zeros(m, N_l_vec(1));
end

% Initialize Cartoon component y_C (calculate initial residual directly from hot-started texture)
y_T_init = mycol2im(D_L * alpha_cells{1}, image_sizes{1}, patch_size);
y_C = y - y_T_init; 

% Step-size memory cache
tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

% Best tracking variables
best_psnr_cartoon = -Inf; best_y_C = [];
best_psnr_texture = -Inf; best_D_L_for_texture = []; best_alpha_for_texture = {};
iterations_without_improvement = 0;

history.psnr_total = zeros(1, MAXITER);
history.psnr_cartoon = zeros(1, MAXITER);
history.psnr_texture = zeros(1, MAXITER);
history.sparsity = zeros(1, MAXITER); % Sparsity record
sumtime = 0;

%% 4a. Real-time Visualization Setup
fprintf('--- 4a. Initializing Visualization Windows ---\n');
hFig1 = figure('Name', 'V-BASTA L0 Real-time Separation', 'Position', [50, 400, 1600, 400]);
hFig2 = figure('Name', 'V-BASTA L0 Learned Dictionary', 'Position', [50, 50, 600, 600]);
hFig3 = figure('Name', 'V-BASTA L0 PSNR Evolution', 'Position', [700, 50, 600, 400]);
hFig4 = figure('Name', 'V-BASTA L0 Sparsity Monitoring', 'Position', [1350, 50, 500, 400]);

%% 4b. V-BASTA (L0) + TV Alternating Optimization Loop
fprintf('--- 4b. Starting V-BASTA (L0) + TV Alternating Optimization Loop ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % STEP 1: Update Texture Component (D_L, alpha) - L0 Logic
    % ==========================================================
    y_target = y - y_C; 
    
    % --- Sub-step A: Update coefficients alpha (L0 Hard Thresholding + TopK) ---
    abs_D = abs(D_L); D_sum = sum(abs_D, 2);
    S_alpha = abs_D' * myim2col(mycol2im(repmat(D_sum, 1, N_l_vec(1)), image_sizes{1}, patch_size), patch_size) + 1e-8; 
    
    [grad_alpha, ~, H_old] = compute_patch_model_gradients(D_L, D_L', alpha_cells{1}, y_target, image_sizes{1}, patch_size);
    tau1_curr = max(gamma1, tau1_history(1) / 1.2);
    
    for k_bt = 1:max_backtrack_iter
        S_effective = tau1_curr * S_alpha;
        v_alpha = alpha_cells{1} - grad_alpha ./ S_effective;
        
        % [Core Modification] L0 Proximal Mapping under Variable Metric
        alpha_candidate = v_alpha;
        
        % 1. Hard Thresholding based on lambda0
        ht_thresh = sqrt(2 * lambda0_penalty ./ S_effective);
        alpha_candidate(abs(alpha_candidate) < ht_thresh) = 0;
        
        % 2. Top-K truncation based on sparsity_k per column
        if sparsity_k < m
            sorted_alpha = sort(abs(alpha_candidate), 1, 'descend');
            k_thresh = sorted_alpha(sparsity_k, :); % K-th largest value per column
            alpha_candidate(abs(alpha_candidate) < k_thresh) = 0; % Zero out elements below threshold
        end
        
        % Check Descent Lemma Upper Bound
        y_recon_T = mycol2im(D_L * alpha_candidate, image_sizes{1}, patch_size);
        f_val = 0.5 * sum((y_recon_T - y_target).^2, 'all');
        Q_val = H_old + compute_descent_lemma_upper_bound_VM(alpha_cells{1}, alpha_candidate, grad_alpha, S_effective);
        
        if f_val <= Q_val
            tau1_history(1) = tau1_curr; break;
        else
            tau1_curr = tau1_curr * tau_bt;
        end
    end
    alpha_cells{1} = alpha_candidate;
    
    % --- Record current sparsity ---
    sparsity_ratio = nnz(alpha_cells{1}) / numel(alpha_cells{1});
    history.sparsity(outerIter) = sparsity_ratio;
    
    % --- Sub-step B: Update Dictionary D_L ---
    [~, grad_D_L, H_old_D] = compute_H_gradients_in_parallel(D_L, alpha_cells, {y_target}, image_sizes, patch_size);
    
    abs_alpha = abs(alpha_cells{1});
    S_D = myim2col(mycol2im(repmat(sum(abs_alpha, 1), d, 1), image_sizes{1}, patch_size), patch_size) * abs_alpha' + 1e-8;
    
    tau2_curr = max(gamma2, tau2_history / 1.2);
    for k_bt = 1:max_backtrack_iter
        S_eff_D = tau2_curr * S_D;
        v_D = D_L - grad_D_L ./ S_eff_D;
        D_candidate = prox_vmetric_L2_ball(v_D, S_eff_D); 
        
        y_recon_T_new = mycol2im(D_candidate * alpha_cells{1}, image_sizes{1}, patch_size);
        f_val = 0.5 * sum((y_recon_T_new - y_target).^2, 'all');
        Q_val = H_old_D + compute_descent_lemma_upper_bound_VM(D_L, D_candidate, grad_D_L, S_eff_D);
        
        if f_val <= Q_val
            tau2_history = tau2_curr; break;
        else
            tau2_curr = tau2_curr * tau_bt;
        end
    end
    D_L = D_candidate;
    
    % ==========================================================
    % STEP 2: Update Cartoon Component y_C (TV Denoising)
    % ==========================================================
    y_T_reconstructed = mycol2im(D_L * alpha_cells{1}, image_sizes{1}, patch_size);
    residual_image = y - y_T_reconstructed;
    
    opts_tv.method = 'l2'; opts_tv.max_itr = 100; opts_tv.tol = 1e-10; opts_tv.rho_r = 1; opts_tv.beta = [1 1 0];
    mu_tv = 1 / zeta;
    tv_out = deconvtv(residual_image, 1, mu_tv, opts_tv);
    y_C = tv_out.f;
    
    % ==========================================================
    % STEP 3: Logging, Best Tracking, and Early Stopping
    % ==========================================================
    elapsed_time = toc; sumtime = sumtime + elapsed_time;
    reconstructed_img = y_T_reconstructed + y_C;
    
    psnr_total = psnr(reconstructed_img, y);
    psnr_texture = psnr(y_T_reconstructed, M_true);
    psnr_cartoon = psnr(y_C, I_true);
    
    history.psnr_total(outerIter) = psnr_total;
    history.psnr_texture(outerIter) = psnr_texture;
    history.psnr_cartoon(outerIter) = psnr_cartoon;
    
    fprintf('Iter %03d/%d: Total=%.2fdB, Txt=%.2fdB, Crt=%.2fdB | Sparsity=%.2f%%, Time=%.2fs\n', ...
            outerIter, MAXITER, psnr_total, psnr_texture, psnr_cartoon, sparsity_ratio*100, elapsed_time);
            
    % --- Best tracking mechanism ---
    improved = false;
    if psnr_cartoon > best_psnr_cartoon
        best_psnr_cartoon = psnr_cartoon; best_y_C = y_C; improved = true;
        fprintf('       *** New optimal Cartoon (PSNR: %.4f dB) ***\n', best_psnr_cartoon);
    end
    if psnr_texture > best_psnr_texture
        best_psnr_texture = psnr_texture; best_D_L_for_texture = D_L; best_alpha_for_texture = alpha_cells; improved = true;
        fprintf('       *** New optimal Texture (PSNR: %.4f dB) ***\n', best_psnr_texture);
    end
    
    if improved, iterations_without_improvement = 0; 
    else, iterations_without_improvement = iterations_without_improvement + 1; 
    end
    
    if iterations_without_improvement >= patience
        fprintf('\n--- No improvement for %d consecutive iterations, triggering L0 early stopping. ---\n', patience);
        history.psnr_total = history.psnr_total(1:outerIter); history.psnr_texture = history.psnr_texture(1:outerIter);
        history.psnr_cartoon = history.psnr_cartoon(1:outerIter); history.sparsity = history.sparsity(1:outerIter);
        break; 
    end

    % ==========================================================
    % STEP 4: Real-time Visualization Updates
    % ==========================================================
    if mod(outerIter, plot_interval) == 0 || outerIter == 1 || outerIter == MAXITER
        if ~isgraphics(hFig1, 'figure'), hFig1 = figure('Name', 'V-BASTA L0 Real-time Separation', 'Position', [50, 400, 1600, 400]); end
        figure(hFig1); subplot(1, 4, 1); imshow(y, []); title('Original Image (y)'); subplot(1, 4, 2); imshow(y_C, []); title(sprintf('Cartoon (y_C)\nPSNR: %.2f dB', psnr_cartoon)); subplot(1, 4, 3); imshow(y_T_reconstructed, []); title(sprintf('Texture (y_T)\nPSNR: %.2f dB', psnr_texture)); subplot(1, 4, 4); imshow(reconstructed_img, []); title(sprintf('Reconstruction (y_T+y_C)\nPSNR: %.2f dB', psnr_total));
        
        if ~isgraphics(hFig2, 'figure'), hFig2 = figure('Name', 'V-BASTA L0 Learned Dictionary', 'Position', [50, 50, 600, 600]); end
        figure(hFig2); 
        if exist('showDictionary', 'file'), showDictionary(D_L); else, imagesc(D_L); colormap gray; end
        title(sprintf('V-BASTA L0 Dictionary (Iter %d)', outerIter));
        
        if ~isgraphics(hFig3, 'figure'), hFig3 = figure('Name', 'V-BASTA L0 PSNR Evolution', 'Position', [700, 50, 600, 400]); end
        figure(hFig3); plot(1:outerIter, history.psnr_total(1:outerIter), 'b-', 'DisplayName', 'PSNR (Total)'); hold on; plot(1:outerIter, history.psnr_texture(1:outerIter), 'r--', 'DisplayName', 'PSNR (Texture)'); plot(1:outerIter, history.psnr_cartoon(1:outerIter), 'g:', 'DisplayName', 'PSNR (Cartoon)'); hold off; xlabel('Iterations'); ylabel('PSNR (dB)'); title('V-BASTA L0 Convergence Curve'); legend show; grid on; xlim([0, MAXITER]);
        
        if ~isgraphics(hFig4, 'figure'), hFig4 = figure('Name', 'V-BASTA L0 Sparsity Monitoring', 'Position', [1350, 50, 500, 400]); end
        figure(hFig4); plot(1:outerIter, history.sparsity(1:outerIter)*100, 'm-', 'LineWidth', 2); xlabel('Iterations'); ylabel('Sparsity (%)'); title('L0 Sparsity Monitoring'); grid on; xlim([0, MAXITER]); ylim([0, 100 * sparsity_k / m * 1.5]);
        
        drawnow;
    end
end
fprintf('\n--- V-BASTA L0 Separation Complete. Total Time: %.2f seconds ---\n', sumtime);

%% 5. Save and Display Optimal Separation Results
fprintf('--- 5. Saving and Displaying the Tracked Optimal Results ---\n');
best_y_T = mycol2im(best_D_L_for_texture * best_alpha_for_texture{1}, image_sizes{1}, patch_size);
reconstructed_best = best_y_C + best_y_T;
psnr_best_total = psnr(reconstructed_best, y);
final_sparsity = nnz(best_alpha_for_texture{1}) / numel(best_alpha_for_texture{1});

save('vbasta_l0_separation_results.mat', ...
     'best_y_C', 'best_y_T', 'best_D_L_for_texture', 'best_alpha_for_texture', ...
     'best_psnr_cartoon', 'best_psnr_texture', 'psnr_best_total', 'final_sparsity');
fprintf('Optimal results saved to vbasta_l0_separation_results.mat.\n');
fprintf('Final L0 Results: Cartoon PSNR = %.4f dB, Texture PSNR = %.4f dB | Final Sparsity = %.2f%%\n', best_psnr_cartoon, best_psnr_texture, final_sparsity*100);

figure('Name', 'Optimal L0 Separation Results (V-BASTA)', 'Position', [100, 200, 1600, 400]);
subplot(1, 4, 1); imshow(y, []); title('Original Image (y)');
subplot(1, 4, 2); imshow(best_y_C, []); title(sprintf('Optimal Cartoon (y_C)\nPSNR: %.2f dB', best_psnr_cartoon));
subplot(1, 4, 3); imshow(best_y_T, []); title(sprintf('Optimal Texture (y_T)\nPSNR: %.2f dB', best_psnr_texture));
subplot(1, 4, 4); imshow(reconstructed_best, []); title(sprintf('Optimal Reconstruction\nPSNR: %.2f dB', psnr_best_total));