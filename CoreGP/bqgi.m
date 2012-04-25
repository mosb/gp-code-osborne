function [ m_Z, sd_Z ] = bqgi( mu, Sigma, l, u, opt )
% Bayesian quadrature for Gaussian integration. Our domain has dimension n.
%
% INPUTS
% mu: mean of Gaussian (n * 1)
% Sigma: covariance of Gaussian (n * n)
% l: vector of lower bounds (n * 1), if missing, assumed to be all -inf
% u: vector of upper bounds (n * 1), if missing, assumed to be all inf
% opt: options (see below)
%
% OUTPUTS
% m_Z: mean of Gaussian integral
% sd_Z: standard deviation of Gaussian integral
%
% Michael Osborne 2012

start_time = cputime;
N = size(mu, 1);

if nargin < 3 || isempty(l)
    l = -inf(N, 1);
end
if nargin < 4 || isempty(u)
    u = inf(N, 1);
end
if nargin < 5
    opt = struct();
end

% Set unspecified fields to default values.
default_opt = struct('total_time', 300);
opt = set_defaults( opt, default_opt );

% =========================================================================
% Possible observations

% slice observations
slices = normcdf(u, mu, diag(Sigma)) - normcdf(l, mu, diag(Sigma));

% gaussian convolution observation
conv = @(m, V) mvnpdf(m, mu, V + Sigma);

% =========================================================================
% Define covariance functions

% covariance function of latent function is normdf(x, x', D);
D = diag(V);

% t: target hyper-rectangle
% s: slice observation
% c: gaussian convolution

u_sub_l = (u - l);
K_tt_vec = -2 * normpdf(0, 0, D) + 2 * normpdf(l, u, D) + ...
    u_sub_l./sqrt(D) .* erf(u_sub_l./(sqrt(2*D)));
K_tt = prod(K_tt_vec);

prod_u_sub_l = prod(u_sub_l);
K_ts = @(i) K_tt_vec(i) * prod_u_sub_l/u_sub_l(i);
K_ss = @(i, j) 
K_sc = @(m, V, i) normcdf(u(i), m(i), V(i, i) + D(i)) - ...
                    normcdf(l(i), m(i), V(i, i) + D(i));
                

while cputime - start_time < opt.total_time
    % take new observation
    
    
end

end



