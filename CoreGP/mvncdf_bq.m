function [int_N, int_x_N] = mvncdf_bq(xl, xu, mu, K)
% Multivariate normal cumulative distribution function (cdf), evaluated
% using Bayesian quadrature.
% int_N = MVNCDF(XL,XU,MU,K) returns the multivariate normal cumulative
%     probability evaluated over the rectangle (hyper-rectangle for D>2) 
%     with lower and upper limits defined by XL and XU, respectively.
% [int_N, int_x_N] = MVNCDF(XL,XU,MU,K) additionally returns the
%     expectation of x over that hyper-rectangle.


 % change coordinates so that the gaussian is aligned with the coordinate
 % axes
 
[M,D] = eig(K);
D(diag_inds(D)) = sqrt(diag(D));