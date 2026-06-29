function y = soft_threshold(x, threshold)
% SOFT_THRESHOLD: Standard soft thresholding operator for L1 regularization.
% y = sign(x) .* max(abs(x) - threshold, 0)
    
    if threshold < 0
        error('The threshold parameter must be a non-negative value.');
    end
    
    y = sign(x) .* max(abs(x) - threshold, 0);
end