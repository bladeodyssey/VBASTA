function D_out = prox_vmetric_L2_ball(V_D, S_D)
% PROX_VMETRIC_L2_BALL: Exact L2-ball proximal mapping under variable metric.
% Solves the true Lagrangian projection optimization problem:
%    arg min_u  0.5 * sum_i S_i * (u_i - v_i)^2
%    s.t.       ||u||_2 <= 1
% 
% Inputs:
%   V_D : Matrix of size [d, m]. The unconstrained intermediate dictionary variable 
%         after the variable-metric gradient step.
%   S_D : Matrix of size [d, m]. The diagonal elements of the variable metric matrix.
%
% Output:
%   D_out : Matrix of size [d, m]. The strictly Lagrangian-projected dictionary atoms.

    [d, m] = size(V_D);
    D_out = V_D; % Default initialization to unconstrained state (mu = 0)

    % 1. Identify which atoms (columns) are outside the unit ball and need projection
    norms = vecnorm(V_D, 2, 1);
    active_cols = norms > 1;

    if ~any(active_cols)
        return; % Return directly if all atoms satisfy the constraint
    end

    % Extract the active columns requiring projection
    X = V_D(:, active_cols);
    A = S_D(:, active_cols);

    % 2. Vectorized Newton's Method to solve the secular equation in parallel
    % Equation: g(mu) = sum_i [ (a_i * x_i) / (a_i + mu) ]^2 - 1 = 0
    % Solve for Lagrange multiplier mu (1 x k row vector, where k is the number of active atoms)

    % Heuristic good initialization: Assuming A is uniform (A \approx mean(A))
    mean_A = mean(A, 1);
    mu = mean_A .* (norms(active_cols) - 1);
    mu = max(mu, 0); % Projection on the sphere necessitates mu > 0

    max_iter = 20; % Newton's method converges rapidly, typically within 3-5 iterations
    tol = 1e-7;    % Error tolerance

    for iter = 1:max_iter
        % Implicit expansion to construct the denominator [d, k]
        A_plus_mu = A + mu; 
        
        % Calculate the corresponding u for the current mu [d, k]
        U = (A .* X) ./ A_plus_mu;
        
        % Evaluate function g(mu) = ||u||^2 - 1
        g_mu = sum(U.^2, 1) - 1;
        
        % Check if max error satisfies the convergence criterion
        if max(abs(g_mu)) < tol
            break;
        end
        
        % Derivative g'(mu) = -2 * sum( (a^2 * x^2) / (a + mu)^3 ) via chain rule
        g_prime_mu = -2 * sum( (A.^2 .* X.^2) ./ (A_plus_mu.^3), 1 );
        
        % Newton's method iteration update
        step = g_mu ./ g_prime_mu;
        mu = mu - step;
        
        % Safety bound: Ensure the multiplier mu remains non-negative
        mu = max(mu, 0);
    end

    % 3. Generate the final exact projected atoms and assign them back
    A_plus_mu_final = A + mu;
    D_out(:, active_cols) = (A .* X) ./ A_plus_mu_final;
end