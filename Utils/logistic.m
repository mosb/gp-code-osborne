function [ f ] = logistic( x, upper, lower )
% logistic function, bounded by lower and upper

if nargin<2
    upper = 1;
end

f = lower + bsxfun(@rdivide,upper-lower,(1+exp(x)));


