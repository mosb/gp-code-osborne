function [tx, gamma] = log_transform(x, gamma)

if nargin < 2
    gamma = (exp(1)-1)^(-1); % numerical scaling factor
end

tx = log(bsxfun(@rdivide, x, gamma) + 1);


