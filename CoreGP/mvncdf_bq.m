function [ m_Z, sd_Z ] = mvncdf_bq( l, u, mu, Sigma, opt )
% Bayesian quadrature for Gaussian integration. Our domain has dimension N.
%
% INPUTS
% mu: mean of Gaussian (N * 1)
% Sigma: covariance of Gaussian (N * N)
% l: vector of lower bounds (N * 1), if missing, assumed to be all -inf
% u: vector of upper bounds (N * 1), if missing, assumed to be all inf
% opt: options (see below)
% in particular, if opt.data is supplied, the locations for the Gaussian
% convolution observations are supplied.
% opt.data(i).m represents the mean of a Gaussian, 
% opt.data(i).m V the diagonal of its diagonal covariance.
% i takes as many sequential values as you want to give.
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
default_opt = struct('total_time', 300, ...
                    'data', []);
opt = set_defaults( opt, default_opt );


% Define terms required to evaluate the covariances between any pair of
% convolution observations, and the covariances between the target
% hyper-rectangle and any convolution observation.
% =========================================================================

% covariance function of latent function is mvnpdf(x, x', diag(D));
D = diag(Sigma);

% prior is mvnpdf(x, mu, diag(L)); this defines the envelope outside which
% we expect our Gaussian integrand to be zero
max_eig = eigs(Sigma, 1);
L = ones(N, 1) * max_eig;

% We're about to compute the product of the bivariate Gaussian cdfs,
% sum_log_mvncdf_lu_N,  that constitutes the self-variance of the target
% hyper-rectangle.

% The two off-diagonal elements of the 2*2 covariance matrix over x1(d) and
% x2(d) (where x1 and x2 are arbitrary points in the domain of integration)
% are equal to M_offdiag(d)
M_offdiag = L.^2./(2*L + D);
% The two on-diagonal elements of the 2*2 covariance matrix over x1(d) and
% x2(d) (where x1 and x2 are arbitrary points in the domain of integration)
% are equal to M_ondiag(d)
M_ondiag = L - M_offdiag;

sum_log_mvncdf_lu_N = 0;
for d = 1:N;
    
    % both variables x1(d) and x2(d) have mu(d) as their mean, l(d) as
    % their lower limit and u(d) as their upper limit
    l_d = [l(d); l(d)];
    u_d = [u(d); u(d)];
    mean_d = [mu(d); mu(d)];
    cov_d = [M_ondiag(d), M_offdiag(d);
            M_offdiag(d), M_ondiag(d)];

    sum_log_mvncdf_lu_N = sum_log_mvncdf_lu_N + ...
        log(mvncdf(l_d, u_d, mean_d, cov_d));
end
% log_variance is the log of the squared output scale, chosen so that K_tt
% is exactly one. Note that we also drop the K_const = N(0,0,2*L + D)
% factor from all covariances; this term gets lumped in with the output
% scale.
log_variance = - sum_log_mvncdf_lu_N;

% K is always a covariance matrix. Below, we use the subscripts:
% t: target hyper-rectangle
% g: new gaussian convolution
% d: all gaussian convolutions

% variance of target hyper-rectangle; we've scaled so that this is one.
K_t = 1;

% Take or read in data
% =========================================================================

% Initialise R_d, the cholesky factor of the covariance matrix over data
R_d = nan(0, 0);
% Initialise D_d = inv(R_d') * (convolution observations);
D_d = nan(0, 1);
% Initialise S_dt = inv(R_d') * K_td';
S_dt = nan(0, 1);

% active_data_selection = true implies that we intelligently select
% observations. If false, we simply read in data from inputs. 
active_data_selection = isempty(opt.data);

% initialise the structure that will store our Gaussian convolution
% observations in it. m represents the mean of such a Gaussian, V the
% diagonal of its diagonal covariance, and conv the actual convolution
% observation. 
data = []; 
    
if active_data_selection
    

    % details yet to be filled in
    
else
    full_data = opt.data;
    
    for d = 1:numel(full_data)
    % add new observation
    
        if rem(d, 10) == 0
            fprintf('\n%g',d);
        else
            fprintf('.');
        end
        
        m_g = full_data(d).m;
        V_g = full_data(d).V;
        
        [R_d, D_d, S_dt, data] = ...
            add_new_datum(m_g, V_g, mu, Sigma, ...
            M_ondiag, M_offdiag, l, u, log_variance, ...
            R_d, D_d, S_dt, data);
    end
end

% Compute final prediction
% =========================================================================
    
[m_Z, sd_Z] = predict(K_t, D_d, S_dt);

end

function [R_d, D_d, S_dt, data] = ...
    add_new_datum(m_g, V_g, mu, Sigma, ...
    M_ondiag, M_offdiag, l, u, log_variance, ...
    R_d, D_d, S_dt, data)
% Update to include new convolution, NB: d includes g
% m_g: mean of new Gaussian convolution (n * 1) 
% V_g: diagonal of (diagonal) covariance of 
%           new Gaussian convolution (n * 1)

num_data = numel(data);

% add new convolution observation to data structure
% =========================================================================
  
data(num_data+1).m = m_g;
data(num_data+1).V = V_g;
data(num_data+1).conv = mvnpdf(m_g, mu, diag(V_g) + Sigma);

num_data = numel(data);
N = length(mu);

% compute new elements of covariance matrix over data, K_d
% =========================================================================
  
log_K_gd = log_variance * ones(1, num_data);
for i = 1:num_data

    m_di = data(i).m;
    V_di = data(i).V;
    
    % K_gd(i) is a product of bivariate Gaussians, one for each dimension
    for d = 1:N;
        
        val_d = [m_g(d); m_di(d)];
        mean_d = [mu(d); mu(d)];
        
        cov_d = [M_ondiag(d) + V_g(d), M_offdiag(d);
                M_offdiag(d), M_ondiag(d) + V_di(d)];
        
        log_K_gd(i) = log_K_gd(i) + ...
            logmvnpdf(val_d, mean_d, cov_d);
    end
end
K_gd = exp(log_K_gd);

% update cholesky factor R_d of covariance matrix over data, K_d
% =========================================================================
  
old_R_d = R_d; % need to store this to update D_d and S_dt, below
K_d = [nan(num_data-1), K_gd(1:end-1)'; 
        K_gd(1:end-1), improve_covariance_conditioning(K_gd(end))];
R_d = updatechol(K_d, old_R_d, num_data);

% update product D_d = inv(R_d') * [data(:).conv]';
% =========================================================================
  
D_d = updatedatahalf(R_d, vertcat(data(:).conv), D_d, old_R_d, num_data);

% compute new elements of covariance vector between target and data
% =========================================================================
  
mean = mu + M_offdiag .* (M_ondiag + V_g).^-1 .* (m_g - mu);
sd = sqrt(M_ondiag - M_offdiag .* (M_ondiag + V_g).^-1 .* M_offdiag);

log_K_tg = log_variance + ...
            sum(lognormpdf(m_g, mu, sqrt(M_ondiag + V_g)) + ...
            truncNormMoments(l, u, mean, sd));
K_tg = exp(log_K_tg);
K_td = [nan(1, num_data-1), K_tg];

% update product S_dt = inv(R_d') * K_td';
% =========================================================================
  
S_dt = updatedatahalf(R_d, K_td', S_dt, old_R_d, num_data);

end

function [m, sd] = predict(K_t, D_d, S_dt)

% gp posterior mean
m = S_dt' * D_d;
%gp posterior variance
var = K_t - S_dt' * S_dt;

sd = sqrt(var);

end
