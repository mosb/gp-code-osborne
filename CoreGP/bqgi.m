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
slices = normcdf(u, mu, diag(V)) - normcdf(l, mu, diag(V));

% observation of convolution with Gaussian mvnpdf(x, m, V)
conv = @(m, V) mvnpdf(m, mu, V + Sigma);

% =========================================================================
% Define covariance functions

% covariance function of latent function is mvnpdf(x, x', diag(D));
D = diag(Sigma);

% prior is mvnpdf(x, mu, diag(L));
L = diag(V);

K_const = prod(2*pi*(2*L + D))^(-0.5);

% we put these quantities into 2 x N shapes, as required by mvnpdf
mu_N = repmat(mu', 2, 1);
l_N = repmat(l', 2, 1);
u_N = repmat(u', 2, 1);

% we put this covariance into 2 x 2 x N shape, as required by mvnpdf
up = @(x) reshape(x,1,1,N);
K_offdiag_N = up(L.^2./(2*L + D));
K_ondiag_N = up(L) - K_offdiag_N;
K_cov_N = [K_ondiag_N, K_offdiag_N; K_offdiag_N, K_ondiag_N];

% prior is mvnpdf(x, mu, diag(L));
L = diag(Sigma);

K_const = prod(2*pi*(2*L + D))^(-0.5);

% we put these quantities into 2 x N shapes, as required by mvnpdf
mu_N = repmat(mu', 2, 1);
l_N = repmat(l', 2, 1);
u_N = repmat(u', 2, 1);

% we put this covariance into 2 x 2 x N shape, as required by mvnpdf
up = @(x) reshape(x,1,1,N);
K_offdiag_N = up(L.^2./(2*L + D));
K_ondiag_N = up(L) - K_offdiag_N;
K_cov_N = [K_ondiag_N, K_offdiag_N; K_offdiag_N, K_ondiag_N];

% t: target hyper-rectangle
% s: slice observation
% g: new gaussian convolution
% h: old gaussian convolutions

K_tt = const * sum(mvncdfN(l_N, u_N, mu_N, sigma

u_sub_l = (u - l);
K_tt_vec = -2 * normpdf(0, 0, D) + 2 * normpdf(l, u, D) + ...
    u_sub_l./sqrt(D) .* erf(u_sub_l./(sqrt(2*D)));
K_tt = prod(K_tt_vec);

% all the required 2D Gaussian integrals between l and u
mvncdf_lu_N = mvncdfN(l_N, u_N, mu_N, K_cov_N);
% all the required 1D Gaussian_integrals between l and u
normcdf_lu_N = normcdf(u, mu, K_ondiag_N(:))';


K_tt = K_const * prod(mvncdf_lu_N);

while cputime - start_time < opt.total_time
    % take new observation
    
    
    best_c
end
    
K_ts = nan(1, N);
for i = 1:N
    K_ts(i) = K_const * mvncdf_lu_N(i) * ...
                        prod(normcdf_lu_N(setdiff(1:end, i)));
end

% the computation got K_ss could be vectorized if it proves excessively
% costly
K_ss = nan(N, N);
for i = 1:N
    for j = 1:N
        if i == j
            K_ss(i, j) = K_const * mvncdf_lu_N(i);
        else
            K_ss(i, j) = K_const * normcdf_lu_N(i) * normcdf_lu_N(j);
        end
    end
end


% while cputime - start_time < opt.total_time
%    % take new observation
% 
% 
%    best_c = 1;
% 
% end

end

function [K_sg, K_gg, K_gh, K_gs] = ...
    convolution_covariances(m_g, V_g, m_h, V_h, mu, ...
    K_const, K_ondiag_N, K_offdiag_N)
% m_g: mean of new Gaussian convolution (n * 1) 
% V_g: diagonal of (diagonal) covariance of 
%           new Gaussian convolution (n * 1)
% m_h: mean of old Gaussian convolutions (n * num_h) 
% V_h: diagonal of (diagonal) covariance of 
%           old Gaussian convolutions (n * num_h)



end