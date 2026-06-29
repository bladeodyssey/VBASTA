function D_out = proxomiga2_2D(D_in)
% PROXOMIGA2_2D: Exact L2 normalization for each column (atom) of a 2D dictionary matrix.
% Ensures that each column d_m of D_out satisfies ||d_m||_2 = 1 (unless d_m is a zero vector).
%
% Input:
%   D_in  : A dictionary matrix of size [d, m].
% Output:
%   D_out : The L2-normalized dictionary matrix.

    % 1. Calculate the L2 norm for each column (atom)
    % vecnorm(D_in, 2, 1) returns a 1 x m row vector containing the norm of each column
    norms = vecnorm(D_in, 2, 1);
    
    % 2. Safety handling: Prevent division by zero
    % Replace zero norms with 1, so a zero vector divided by 1 remains a zero vector
    norms(norms == 0) = 1;
    
    % 3. Execute normalization
    % Divide each column by its own norm. MATLAB's implicit expansion handles this automatically.
    D_out = D_in ./ norms;

end