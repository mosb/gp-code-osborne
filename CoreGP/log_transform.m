function [tx, gamma] = log_transform(x)

gamma = (exp(1)-1)^(-1); % numerical scaling factor

tx = log(bsxfun(@rdivide, x, gamma) + 1);


