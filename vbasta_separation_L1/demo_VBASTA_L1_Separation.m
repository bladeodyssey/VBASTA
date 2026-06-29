%% Cartoon-Texture Separation using V-BASTA Framework (L1 Pre-training with Best Tracking)
% =========================================================================
% Features:
% 1. Variable Metric Optimization: Accelerates the learning of the texture 
%    components (D_L, alpha) via spatially adaptive metrics.
% 2. Structural Decomposition: Couples with TV Denoising to extract the 
%    piecewise-smooth cartoon component (y_C).
% 3. Intelligent Best Tracking: Independently tracks the highest PSNR for 
%    cartoon and texture, saving the optimal state dynamically.
% 4. Early Stopping: Implements a patience mechanism to halt iterations 
%    when no further structural improvements are observed.
% 5. Real-time Monitoring: Three-panel visualization of the separation progress.
%revised 2026.6.17 Minghui Xu
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. Parameter Settings
fprintf('--- 1. Parameter Settings (V-BASTA L1 Separation) ---\n');
m = 100;              % Number of dictionary atoms
patch_size = 11;      % Spatial filter size
lambda1 = 0.1;        % L1 sparsity weight for texture
zeta = 0.025;         % TV smoothness weight for cartoon

% --- V-BASTA Optimization Parameters ---
MAXITER = 1000;       
max_backtrack_iter = 30; 
gamma1 = 0.006;       % Initial scaling for S_alpha
gamma2 = 0.006;       % Initial scaling for S_D
tau_bt = 1.5;         % Backtracking decay multiplier

% --- Visualization & Early Stopping Parameters ---
plot_interval = 5;    % Update plots every N iterations
patience = 30;       % Stop if no PSNR improvement for 30 consecutive iterations

%% 2. Load and Synthesize Data
fprintf('--- 2. Loading Data ---\n');
% Assuming the files contain 'imgc' (Cartoon) and 'fense2' (Texture)
load('datasets/separation/cat4.mat'); 
load('datasets/separation/fense2.mat'); 

I_true = im2double(imgc); 
M_true = 0.4 * im2double(imresize(fense2, [256, 256]));
y = I_true + M_true;  % Synthesize observed composite image
numpic = 1; 

image_sizes{1} = size(y);
dummy_patches = myim2col(zeros(image_sizes{1}), patch_size);
N_l_vec(1) = size(dummy_patches, 2);

%% 3. Initialization and Memory Cache
fprintf('--- 3. Initializing Variables ---\n');
d = patch_size^2;
D_L = randn(d, m); 
D_L = proxomiga2_2D(D_L);
alpha_cells{1} = zeros(m, N_l_vec(1));
y_C = zeros(size(y)); 

% Step-size memory cache (Warm Start)
tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

% --- Best Tracking Variables ---
best_psnr_cartoon = -Inf; best_y_C = [];
best_psnr_texture = -Inf; best_D_L_for_texture = []; best_alpha_for_texture = {};
iterations_without_improvement = 0;

history.psnr_total = zeros(1, MAXITER);
history.psnr_cartoon = zeros(1, MAXITER);
history.psnr_texture = zeros(1, MAXITER);
sumtime = 0;

%% 4a. Real-time Visualization Setup
fprintf('--- 4a. Initializing Visualization Windows ---\n');
hFig1 = figure('Name', 'V-BASTA Real-time Separation', 'Position', [50, 400, 1600, 400]);
hFig2 = figure('Name', 'V-BASTA Learned Dictionary', 'Position', [50, 50, 600, 600]);
hFig3 = figure('Name', 'V-BASTA PSNR Evolution', 'Position', [700, 50, 600, 400]);

%% 4b. V-BASTA + TV Alternating Optimization Loop
fprintf('--- 4b. Starting V-BASTA + TV Alternating Optimization Loop ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % STEP 1: Update Texture Component (D_L, alpha) via V-BASTA
    % ==========================================================
    y_target = y - y_C; % The target for texture is the residual
    
    % --- Cache S_alpha (Variable Metric Matrix) ---
    abs_D = abs(D_L);
    D_sum = sum(abs_D, 2);
    D_rep = repmat(D_sum, 1, N_l_vec(1));
    Img_overlap = mycol2im(D_rep, image_sizes{1}, patch_size);
    P_overlap = myim2col(Img_overlap, patch_size);
    S_alpha = abs_D' * P_overlap + 1e-8; 
    
    % --- Sub-step A: Update sparse coefficients (alpha) ---
    [grad_alpha, ~, H_old] = compute_patch_model_gradients(D_L, D_L', alpha_cells{1}, y_target, image_sizes{1}, patch_size);
    tau1_curr = max(gamma1, tau1_history(1) / 1.2);
    
    for k_bt = 1:max_backtrack_iter
        S_effective = tau1_curr * S_alpha;
        v_alpha = alpha_cells{1} - grad_alpha ./ S_effective;
        alpha_candidate = soft_threshold(v_alpha, lambda1 ./ S_effective);
        
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
    
    % Failure handling for sparse coefficients update
    if k_bt == max_backtrack_iter
        warning('Maximum backtracking steps reached for sparse coefficient update. Descent condition not strictly met.');
    end
    
    alpha_cells{1} = alpha_candidate;
    
    % --- Sub-step B: Update Dictionary (D_L) ---
    [~, grad_D_L, H_old_D] = compute_H_gradients_in_parallel(D_L, alpha_cells, {y_target}, image_sizes, patch_size);
    
    % Calculate S_D
    abs_alpha = abs(alpha_cells{1});
    X_mat = repmat(sum(abs_alpha, 1), d, 1);
    S_D = myim2col(mycol2im(X_mat, image_sizes{1}, patch_size), patch_size) * abs_alpha' + 1e-8;
    
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
    
    % Failure handling for Dictionary update
    if k_bt == max_backtrack_iter
        warning('Maximum backtracking steps reached for Dictionary update.');
    end
    
    D_L = D_candidate;
    
    % ==========================================================
    % STEP 2: Update Cartoon Component y_C (TV Denoising)
    % ==========================================================
    y_T_reconstructed = mycol2im(D_L * alpha_cells{1}, image_sizes{1}, patch_size);
    residual_image = y - y_T_reconstructed;
    
    % TV Denoising configurations
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
    
    fprintf('Iter %04d/%d: PSNR(Total)=%.2f, PSNR(Texture)=%.2f, PSNR(Cartoon)=%.2f, Time=%.2fs\n', ...
            outerIter, MAXITER, psnr_total, psnr_texture, psnr_cartoon, elapsed_time);
            
    % --- Best tracking mechanism (Excluding the unstable 1st iteration) ---
    improved = false;
    
    if outerIter > 10
        if psnr_cartoon > best_psnr_cartoon
            best_psnr_cartoon = psnr_cartoon;
            best_y_C = y_C;
            improved = true;
            fprintf('       *** Found new optimal Cartoon result (PSNR: %.4f dB) ***\n', best_psnr_cartoon);
        end
        if psnr_texture > best_psnr_texture
            best_psnr_texture = psnr_texture;
            best_D_L_for_texture = D_L;
            best_alpha_for_texture = alpha_cells;
            improved = true;
            fprintf('       *** Found new optimal Texture result (PSNR: %.4f dB) ***\n', best_psnr_texture);
        end
        
        % --- Early stopping check ---
        if improved
            iterations_without_improvement = 0; 
        else
            iterations_without_improvement = iterations_without_improvement + 1; 
        end
        
        if iterations_without_improvement >= patience
            fprintf('\n--- No improvement for %d consecutive iterations. Triggering early stopping. ---\n', patience);
            history.psnr_total = history.psnr_total(1:outerIter);
            history.psnr_texture = history.psnr_texture(1:outerIter);
            history.psnr_cartoon = history.psnr_cartoon(1:outerIter);
            break; 
        end
    else
        fprintf('       (Iteration 10 metrics logged as baseline, excluded from best tracking)\n');
    end

    % ==========================================================
    % STEP 4: Real-time Visualization Updates
    % ==========================================================
    if mod(outerIter, plot_interval) == 0 || outerIter == 1 || outerIter == MAXITER
        
        % Figure 1: Separation Results
        if ~isgraphics(hFig1, 'figure'), hFig1 = figure('Name', 'V-BASTA Real-time Separation', 'Position', [50, 400, 1600, 400]); end
        figure(hFig1); 
        subplot(1, 4, 1); imshow(y, []); title('Original Composite (y)');
        subplot(1, 4, 2); imshow(y_C, []); title(sprintf('Cartoon (y_C)\nPSNR: %.2f dB', psnr_cartoon));
        subplot(1, 4, 3); imshow(y_T_reconstructed, []); title(sprintf('Texture (y_T)\nPSNR: %.2f dB', psnr_texture));
        subplot(1, 4, 4); imshow(reconstructed_img, []); title(sprintf('Reconstruction (y_T+y_C)\nPSNR: %.2f dB', psnr_total));
        
        % Figure 2: Dictionary
        if ~isgraphics(hFig2, 'figure'), hFig2 = figure('Name', 'V-BASTA Learned Dictionary', 'Position', [50, 50, 600, 600]); end
        figure(hFig2); 
        if exist('showDictionary', 'file')
            showDictionary(D_L);
            title(sprintf('V-BASTA Learned Dictionary (Iter %d)', outerIter));
        else
            imagesc(D_L); colormap gray; title(sprintf('Dictionary Matrix (Iter %d)', outerIter));
        end
        
        % Figure 3: PSNR Curves
        if ~isgraphics(hFig3, 'figure'), hFig3 = figure('Name', 'V-BASTA PSNR Evolution', 'Position', [700, 50, 600, 400]); end
        figure(hFig3); 
        plot(1:outerIter, history.psnr_total(1:outerIter), 'b-', 'DisplayName', 'PSNR (Total)'); hold on;
        plot(1:outerIter, history.psnr_texture(1:outerIter), 'r--', 'DisplayName', 'PSNR (Texture)');
        plot(1:outerIter, history.psnr_cartoon(1:outerIter), 'g:', 'DisplayName', 'PSNR (Cartoon)'); hold off;
        xlabel('Iterations'); ylabel('PSNR (dB)'); title('V-BASTA Real-time PSNR Evolution');
        legend show; grid on; xlim([0, MAXITER]);
        
        drawnow;
    end
end
fprintf('\n--- V-BASTA Separation Complete. Total Time: %.2f seconds ---\n', sumtime);

%% 5. Save and Display Optimal Separation Results
fprintf('--- 5. Saving and Displaying the Tracked Optimal Results ---\n');

% Reconstruct the optimal composite result
best_y_T = mycol2im(best_D_L_for_texture * best_alpha_for_texture{1}, image_sizes{1}, patch_size);
reconstructed_best = best_y_C + best_y_T;
psnr_best_total = psnr(reconstructed_best, y);

% Save optimal state securely
SAVE_FILE = 'vbasta_l1_separation_results.mat';
save(SAVE_FILE, 'best_y_C', 'best_y_T', 'best_D_L_for_texture', 'best_alpha_for_texture', ...
     'best_psnr_cartoon', 'best_psnr_texture', 'psnr_best_total', 'y');

fprintf('Optimal L1 results successfully saved to %s.\n', SAVE_FILE);
fprintf('Final Results: Optimal Cartoon PSNR = %.4f dB, Optimal Texture PSNR = %.4f dB\n', best_psnr_cartoon, best_psnr_texture);
fprintf('You may now run the L0 script for rapid hot-start fine-tuning.\n');

% Display the final optimal separation
figure('Name', 'Optimal Separation Results (V-BASTA L1)', 'Position', [100, 200, 1600, 400]);
subplot(1, 4, 1); imshow(y, []); title('Original Composite (y)');
subplot(1, 4, 2); imshow(best_y_C, []); title(sprintf('Optimal Cartoon (y_C)\nPSNR: %.2f dB', best_psnr_cartoon));
subplot(1, 4, 3); imshow(best_y_T, []); title(sprintf('Optimal Texture (y_T)\nPSNR: %.2f dB', best_psnr_texture));
subplot(1, 4, 4); imshow(reconstructed_best, []); title(sprintf('Optimal Reconstruction\nPSNR: %.2f dB', psnr_best_total));