function result = compute_descent_lemma_upper_bound_VM(x_old, x_new, grad_H_at_x_old, S_metric)
% COMPUTE_DESCENT_LEMMA_UPPER_BOUND_VM: Calculates the right-hand side upper bound Q 
% of the descent lemma under a variable metric.
%
% Q(x_new, x_old) = <x_new - x_old, grad_H> + (1/2) * ||x_new - x_old||_{S_metric}^2
%
% Inputs:
%   x_old           : The previous iteration point.
%   x_new           : The new candidate iteration point.
%   grad_H_at_x_old : The gradient evaluated at the previous point.
%   S_metric        : The variable metric matrix for the current iteration step 
%                     (same dimensions as x_old).
%
% Output:
%   result          : The differential quadratic majorizer value. 
%                     Note: Add H(x_old) to this result to get the full Q(x_new, x_old).

    delta_x = x_new - x_old;
    
    % 1. Calculate the inner product term: <x_new - x_old, grad_H>
    inner_product = sum(delta_x .* grad_H_at_x_old, 'all');
    
    % 2. Calculate the squared weighted L2 norm: sum( S .* (delta_x)^2 )
    norm_squared_weighted = sum(S_metric .* (delta_x.^2), 'all');
    
    result = inner_product + 0.5 * norm_squared_weighted;
end