function [tx, gamma] = log_transform(x, gamma)

tx = log(bsxfun(@rdivide, x, gamma) + 1);


