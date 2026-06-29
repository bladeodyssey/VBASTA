function [grad_alpha_H_cells, grad_D_H_total, total_g_y] = compute_H_gradients_in_parallel(D_L, alpha_cells, y_cells, image_sizes, patch_size)
% COMPUTE_H_GRADIENTS_IN_PARALLEL: Computes the gradients of H and reconstruction error for multiple images.
%
% This function iterates through the cell array of images, calling the single-image 
% gradient computation function for each. It correctly accumulates the gradient
% with respect to the shared dictionary D_L and calculates the total reconstruction error.

    numpic = length(y_cells);
    grad_D_H_total = zeros(size(D_L));
    grad_alpha_H_cells = cell(1, numpic);
    total_g_y = 0;

    % Precompute the transpose of the dictionary (outside the loop to save time)
    D_L_t = D_L';
    
    % A 'parfor' loop can be used here for parallel acceleration if the Parallel Computing Toolbox is available
    for l = 1:numpic
        % Call the single-image gradient computation function for the l-th image
        [grad_alpha_H_cells{l}, grad_D_H_l, g_y_l] = compute_patch_model_gradients(D_L, D_L_t, alpha_cells{l}, y_cells{l}, image_sizes{l}, patch_size);
        
        % Accumulate the dictionary gradient and reconstruction error
        grad_D_H_total = grad_D_H_total + grad_D_H_l;
        total_g_y = total_g_y + g_y_l;
    end
end