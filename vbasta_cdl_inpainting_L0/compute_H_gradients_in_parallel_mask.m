function [grad_alpha_H_cells, grad_D_H_total, total_g_y] = compute_H_gradients_in_parallel_mask(D_L, alpha_cells, y_masked_cells, M_cells, image_sizes, patch_size)
% COMPUTE_H_GRADIENTS_IN_PARALLEL_MASK 
% Computes the gradients of H and reconstruction error for multiple corrupted images (with masks) in parallel.
%
%   This function uses a parfor loop to call the masked single-image gradient computation
%   for each image in the cell array. It securely accumulates the gradient with respect 
%   to the shared dictionary D_L and calculates the overall effective data fidelity error (only where M=1).

    numpic = length(y_masked_cells);
    
    % Initialize reduction variables for the parfor loop
    grad_D_H_total = zeros(size(D_L));
    grad_alpha_H_cells = cell(1, numpic);
    total_g_y = 0;

    % Precompute the transpose of the dictionary (outside the loop to save time)
    D_L_t = D_L';
    
    for l = 1:numpic
        % Extract local variables for the current image to avoid broadcasting the entire cell array
        alpha_l = alpha_cells{l};
        y_masked_l = y_masked_cells{l};
        M_l = M_cells{l};
        img_size_l = image_sizes{l};
        
        % Call the masked single-image gradient computation function
        [grad_alpha_l, grad_D_H_l, g_y_l] = compute_patch_model_gradients_mask(...
            D_L, D_L_t, alpha_l, y_masked_l, M_l, img_size_l, patch_size);
        
        % Save the gradient with respect to the sparse coefficients alpha for the current image
        grad_alpha_H_cells{l} = grad_alpha_l;
        
        % Accumulate the dictionary gradient and reconstruction error (supported by parfor reduction)
        grad_D_H_total = grad_D_H_total + grad_D_H_l;
        total_g_y = total_g_y + g_y_l;
    end
end