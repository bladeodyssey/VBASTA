function [grad_alpha, grad_D, g_y] = compute_patch_model_gradients_mask(D_L, D_L_t, alpha, y_masked, M, image_size, patch_size)
% COMPUTE_PATCH_MODEL_GRADIENTS_MASK 
% Computes the gradients and reconstruction error for the Image Inpainting task.

    % Ensure the input is in full format (non-sparse) for subsequent dense calculations
    if issparse(alpha)
        alpha = full(alpha);
    end

    % Step 1: Reconstruct all image patches independently
    reconstructed_patches = D_L * alpha;

    % Step 2: Superimpose and synthesize the complete reconstructed image
    y_recon = mycol2im(reconstructed_patches, image_size, patch_size);

    % Step 3: Calculate the masked reconstruction error g(y) (only compute error on observed pixels)
    residual = M .* (y_recon - y_masked);
    g_y = 0.5 * sum(residual(:).^2);

    % Step 4: Extract residual patches for each spatial position from the residual image
    residual_patches_from_image = myim2col(residual, patch_size);

    % Step 5: Compute time-domain gradients
    % 5.1 Compute the gradient with respect to the sparse coefficients (alpha)
    grad_alpha = D_L_t * residual_patches_from_image;
    
    % 5.2 Compute the gradient with respect to the dictionary (D_L)
    grad_D = residual_patches_from_image * alpha';

end