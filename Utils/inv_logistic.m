function [ x ] = inv_logistic( f, upper )
% logistic function

ee = min(eps, upper/10);

if nargin<2
    upper = 1;
end

x = -log(bsxfun(@rdivide, upper - 2*ee, f - ee) - 1);


