function [ f ] = logistic( x, upper )
% logistic function

ee = min(eps, upper/10);

if nargin<2
    upper = 1;
end

f = ee + bsxfun(@rdivide,upper-2*ee,(1+exp(-x)));


