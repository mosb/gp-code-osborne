function [log_mean_evidence, log_var_evidence, ev_params, del_gp_hypers_SE] = ...
    log_evidence(samples, prior, ...
    l_gp_hypers_SE, tl_gp_hypers_SE, del_gp_hypers_SE, opt)
% [log_mean_evidence, log_var_evidence, ev_params] = ...
%     log_evidence(samples, prior, ...
%     l_gp_hypers, tl_gp_hypers, del_gp_hypers, opt)
%
% Returns the log of the mean-evidence, the log of the variance of the
% evidence, and a  a structure ev_params to ease its future computation.
%
% OUTPUTS
% - log_mean_evidence
% - log_var_evidence
% - ev_params: (see expected_uncertainty_evidence.m) has fields
%   'x_c
%   'sqd_dist_stack_s
%   'R_tl_s
%   'K_tl_s
%   'inv_K_tl_s
%   'jitters_l
%   'sqd_dist_stack_s
%   'R_del
%   'K_del
%   'ups_l
%   'ups_del
%   'Ups_sc_s
%   'del_inv_K_del
%   'delta_tl_sc
%   'minty_del
%   'log_mean_second_moment
%
% INPUTS
% - samples: requires fields
%   'locations
%   'log_l
% - prior requires fields
%   'mean
%   'covariance
% - l_gp_hypers_SE: hypers for sqd exp covariance over l, with fields
%   * log_output_scale
%   * log_input_scales
% - tl_gp_hypers_SE: hypers for sqd exp covariance over tl, with fields
%   * log_output_scale
%   * log_input_scales
% - del_gp_hypers_SE: hypers for sqd exp covariance over del, with fields
%   * log_output_scale
%   * log_input_scales

% Load options, set to default if not available
% ======================================================

no_ev_params = nargin<3;
if nargin<4
    opt = struct();
end

default_opt = struct('num_c', 200,... % number of candidate points
                     'num_box_scales', 5, ... % defines the box over which to take candidates
                     'allowed_cond_error',10^-14, ... % allowed conditioning error
                     'sds_tl_log_input_scales', false, ... % sds_tl_log_input_scales represents the posterior standard deviations in the input scales for tr. If false, a delta function posterior is assumed. 
                     'gamma', 100, ... % log_transform scaling factor.
                     'plots', false ...   % plot transformed surfaces.
                        );
opt = set_defaults( opt, default_opt );

num_samples = size(samples.locations, 1);

% The number of candidate locations to sample.
opt.num_c = max(opt.num_c, samples.locations);

% candidate locations will be constrained to a box defined by the prior
lower_bound = prior.mean - opt.num_box_scales*sqrt(diag(prior.covariance))';
upper_bound = prior.mean + opt.num_box_scales*sqrt(diag(prior.covariance))';

% candidate locations are taken to be as far away from each other and
% existing sample locations as possible, according to a Mahalanobis
% distance with diagonal covariance matrix with diagonal defined as
% mahal_scales.
mahal_scales = exp(l_gp_hypers_SE.log_input_scales);

% find the candidate locations, far removed from existing samples, with the
% use of a Voronoi diagram
x_c = find_farthest(samples.locations, ...
                    [lower_bound; upper_bound], opt.num_c, ...
                     mahal_scales);

% the combined sample locations
x_sc = [samples.locations; x_c];
[num_sc, D] = size(x_sc);

% rescale all log-likelihood values for numerical accuracy; we'll correct
% for this at the end of the function
l_s = samples.scaled_l;

% opt.gamma is corrected for after l_s has already been divided by
% exp(samples.max_log_l_s). tl_s is its correct value, but log(opt.gamma) has
% effectively had samples.max_log_l_s subtracted from it. 
tl_s = samples.tl;


% Compute our covariance matrices and their cholesky factors
% ======================================================

% input hyperparameters are for a sqd exp covariance, whereas in all that
% follows we use a gaussian covariance. We correct the output scales
% appropriately.
l_gp_hypers = sqdexp2gaussian(l_gp_hypers_SE);
tl_gp_hypers = sqdexp2gaussian(tl_gp_hypers_SE);
if ~isempty(del_gp_hypers_SE)
    del_gp_hypers = del_gp_hypers_SE;
end
% otherwise, we'll work these out below.


% squared distances are expensive to compute, so we store them for use in
% the functions below, rather than having each function compute them
% afresh.
sqd_dist_stack_sc = sqd_dist_stack(x_sc,x_sc);
sqd_dist_stack_s = sqd_dist_stack_sc(1:num_samples, 1:num_samples, :);
sqd_dist_stack_s_sc = sqd_dist_stack_sc(1:num_samples, :, :);

% The gram matrix over the likelihood, its cholesky factor, and the
% product of the precision and the data
K_l = gaussian_mat(sqd_dist_stack_s, l_gp_hypers);
K_l = improve_covariance_conditioning(K_l, ...
    l_s, ...
    opt.allowed_cond_error);
R_l = chol(K_l);
inv_K_l_l = solve_chol(R_l, l_s);
% The covariance over the likelihood between x_sc and x_s
K_l_sc = gaussian_mat(sqd_dist_stack_s_sc, l_gp_hypers);

% The gram matrix over the transformed likelihood, its cholesky factor, and
% the product of the precision and the data
K_tl = gaussian_mat(sqd_dist_stack_s, tl_gp_hypers);
[K_tl, jitters_tl] = improve_covariance_conditioning(K_tl, ...
    tl_s, ...
    opt.allowed_cond_error);
R_tl = chol(K_tl);
inv_K_tl_tl = solve_chol(R_tl, tl_s);
% The covariance over the transformed likelihood between x_sc and x_s
K_tl_sc = gaussian_mat(sqd_dist_stack_s_sc, tl_gp_hypers);

% Compute delta, the difference between the mean of the transformed (log)
% likelihood and the transform of the mean likelihood
% ======================================================

% the mean of the likelihood at x_sc
mean_l_sc =  K_l_sc' * inv_K_l_l;
% use a crude thresholding here as our tilde transformation will fail if
% the mean goes below zero
mean_l_sc = max(mean_l_sc, eps);

% the mean of the transformed (log) likelihood at x_sc
mean_tl_sc = K_tl_sc' * inv_K_tl_tl;

% the difference between the mean of the transformed (log) likelihood and
% the transform of the mean likelihood
delta_tl_sc = mean_tl_sc - log_transform(mean_l_sc, opt.gamma);


if isempty(del_gp_hypers_SE);
    fprintf('Fitting GP to delta-observations...\n');

    % Set up GP.
    gp_hypers_del.mean = [];
    gp_hypers_del.lik = log(0.01);  % Values go between 0 and 1, so no need to scale.
    init_lengthscales = tl_gp_hypers_SE.log_input_scales - log(10);
    init_output_variance = tl_gp_hypers_SE.log_output_scale;
    gp_hypers_del.cov = [init_lengthscales(1) init_output_variance];
    init_hypers = gp_hypers_del;
    inference = @infExact;
    likfunc = @likGauss;
    meanfunc = {'meanZero'};
    max_iters = 100;
    covfunc = @covSEiso;

    gp_hypers_del = minimize(gp_hypers_del, @gp_fixedlik, -max_iters, ...
                             inference, meanfunc, covfunc, likfunc, ...
                             x_sc, delta_tl_sc);        
                         
    if any(isnan(gp_hypers_del.cov))
        gp_hypers_del = init_hypers;
        warning('Optimizing hypers on delta failed');
    end                         

    del_gp_hypers_SE.log_output_scale = gp_hypers_del.cov(end);
    del_gp_hypers_SE.log_input_scales(1:D) = gp_hypers_del.cov(1:end - 1);
    del_gp_hypers = sqdexp2gaussian(del_gp_hypers_SE);
end


% Some debugging plots.
if opt.plots && D == 1
    log_ev_plots(x_sc, mean_l_sc, mean_tl_sc, delta_tl_sc, ...
        l_gp_hypers, tl_gp_hypers, del_gp_hypers, samples, opt); 
end

% Compute various quantities required to evaluate the mean evidence
% ======================================================

% squared distances are expensive to compute, so we store them for use in
% the functions below, rather than having each funciton compute them
% afresh.
[sqd_x_sub_mu_stack_sc, x_sub_mu_stack_sc] = ...
    sqd_dist_stack(x_sc, prior.mean);
sqd_x_sub_mu_stack_s = sqd_x_sub_mu_stack_sc(1:num_samples, :, :);
x_sub_mu_stack_s = x_sub_mu_stack_sc(1:num_samples, :, :);

% calculate ups for the likelihood, where ups is defined as
% ups_s = int K(x, x_s)  priol(x) dx
ups_l = small_ups_vec(sqd_x_sub_mu_stack_s, l_gp_hypers, prior);

% calculate ups for the likelihood, where ups is defined as
% ups_s = int K(x, x_s)  priol(x) dx
ups_tl = small_ups_vec(sqd_x_sub_mu_stack_s, tl_gp_hypers, prior);


% compute mean of int l(x) p(x) dx given l_s
ups_inv_K_l = solve_chol(R_l, ups_l)';
minty_l = ups_inv_K_l * l_s;

% calculate Ups for delta & the likelihood, where Ups is defined as
% Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
Ups_del_l = big_ups_mat...
    (sqd_x_sub_mu_stack_sc, sqd_x_sub_mu_stack_s, ...
    tr(sqd_dist_stack_s_sc), ...
    del_gp_hypers, l_gp_hypers, prior);

% The gram matrix over delta and its cholesky factor
K_del = gaussian_mat(sqd_dist_stack_sc, del_gp_hypers);
% The candidate locations are more important to preserve noise-free than
% the sample locatios, at which delta is equal to the prior mean anyway
% (zero)
importance_sc = ones(num_sc,1);
importance_sc(num_samples + 1 : end) = 2;
K_del = improve_covariance_conditioning(K_del, importance_sc, ...
    opt.allowed_cond_error);
R_del = chol(K_del);     

% compute mean of int delta(x) l(x) p(x) dx given l_s and delta_tl_sc
del_inv_K_del = solve_chol(R_del, delta_tl_sc)';
Ups_inv_K_del_l = solve_chol(R_l, Ups_del_l')';
minty_del_l = del_inv_K_del * Ups_inv_K_del_l * l_s;

% calculate ups for delta, where ups is defined as
% ups_s = int K(x, x_s)  priol(x) dx
ups_del = small_ups_vec(sqd_x_sub_mu_stack_sc, del_gp_hypers, prior);

% compute mean of int delta(x) p(x) dx given l_s and delta_tl_sc 
ups_inv_K_del = solve_chol(R_del, ups_del)';
minty_del = ups_inv_K_del * delta_tl_sc;

% the correction factor due to l being non-negative
correction = minty_del_l + opt.gamma * minty_del;

% the mean evidence
mean_ev = minty_l + correction;

% mean_ev has been determined using the rescaled log-likelihoods (that have
% had the maximum log likelihood subtracted off), we return correct values
% by scaling back again)
log_mean_evidence = samples.max_log_l + log(mean_ev);

% Compute the further terms required to determine the variance in the
% evidence
% ======================================================

% calculate ups2 for the likelihood, where ups2 is defined as
% ups2_s = int int K(x, x') K(x', x_s) priol(x) prior(x') dx dx'
ups2_l = small_ups2_vec(sqd_x_sub_mu_stack_s, tl_gp_hypers, ...
    l_gp_hypers, prior);  

% calculate chi for the likelihood, where chi is defined as
% chi = int int K(x, x') priol(x) prior(x') dx dx'
chi_tl = small_chi_const(tl_gp_hypers, prior);
      
% calculate Chi for the likelihood, where Chi is defined as
% Chi_l = int int K(x_s, x) K(x, x') K(x', x_s) priol(x)
% prior(x') dx dx'
Chi_l_tl_l = big_chi_mat(sqd_x_sub_mu_stack_s, sqd_dist_stack_s, ...
    l_gp_hypers, tl_gp_hypers, prior);

% calculate Ups for the likelihood and the likelihood, where Ups is defined as 
% Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
Ups_tl_l = big_ups_mat...
    (sqd_x_sub_mu_stack_s, sqd_x_sub_mu_stack_s, ...
    sqd_dist_stack_s, ...
    tl_gp_hypers, l_gp_hypers, prior);

% compute the variance of int log_transform(l)(x) p(x) dx given l_s
ups_inv_K_tl = solve_chol(R_tl, ups_tl)';
Vinty_tl = chi_tl - ups_inv_K_tl * ups_tl;

% compute int dx p(x) int dx' p(x') C_(tl|s)(x, x') m_(l|s)(x')
Ups_inv_K_tl_l = Ups_tl_l * inv_K_l_l;
Cminty_tl_l = ups2_l' * inv_K_l_l ...
    - ups_inv_K_tl * Ups_inv_K_tl_l;

% compute int dx p(x) int dx' p(x') m_(l|s)(x) C_(tl|s)(x, x') m_(l|s)(x')
inv_R_Ups_inv_K_tl_l = R_tl'\Ups_inv_K_tl_l;
mCminty_l_tl_l = inv_K_l_l' * Chi_l_tl_l * inv_K_l_l ...
            - sum(inv_R_Ups_inv_K_tl_l.^2);

% variance of the evidence
var_ev = opt.gamma^2 * Vinty_tl ...
    + 2 * opt.gamma * Cminty_tl_l ...
    + mCminty_l_tl_l;


if opt.sds_tl_log_input_scales
    % we account for our uncertainty in the log input scales
    
    % the variances of our posteriors over our input scales. We assume the
    % covariance matrix has zero off-diagonal elements; the posterior is
    % spherical. 
    V_theta = opt.sds_tl_log_input_scales.^2;
    if size(V_theta,1) == 1
        V_theta = V_theta';
    end   
    
    % compute mean of int tl(x) l(x) p(x) dx given l_s and tl_s 
    minty_tl_l = inv_K_tl_tl' * Ups_inv_K_tl_l;
    
    % compute mean of int tl(x) p(x) dx given l_s and tl_s 
    minty_tl = ups_tl' * inv_K_tl_tl;
    
    % hyperparameters for gp over the transformed likelihood, tl, assumed
    % to have zero mean
    tl_input_scales = exp(tl_gp_hypers.log_input_scales);
    inv_sqd_tl_input_scales_stack = ...
        reshape(tl_input_scales.^-2, 1, 1, D);
    
    % Dtheta_K_tl_a_s is the gradient of the Gaussian covariance over the
    % transformed likelihood between x_a and x_s: each plate in the stack
    % is the derivative with respect to a different log input scale
    Dtheta_K_tl_const = -1 + bsxfun(@times, ...
        sqd_dist_stack_s, ...
        inv_sqd_tl_input_scales_stack);
    Dtheta_K_tl = bsxfun(@times, K_tl, Dtheta_K_tl_const);
    
    % Dtheta_Ups_tl_l is the modification of Ups_tl_l to allow for
    % derivatives wrt log input scales: each plate in the stack is the
    % derivative with respect to a different log input scale.
    [Dtheta_Ups_tl_l_const, Dtheta_ups_tl_const] = DTheta_consts...
        (inv_sqd_tl_input_scales_stack, x_sub_mu_stack_s, ...
        prior, tl_gp_hypers, l_gp_hypers);
    Dtheta_Ups_tl_l = bsxfun(@times, Ups_tl_l, Dtheta_Ups_tl_l_const);
    
    % Dtheta_ups_tl is the modification of ups_tl to allow for
    % derivatives wrt log input scales: each plate in the stack is the
    % derivative with respect to a different log input scale.
    Dtheta_ups_tl = bsxfun(@times, ups_tl, Dtheta_ups_tl_const);
        
    int_ml_Dtheta_mtl = bsxfun(@plus, ...
            - minty_tl_l - opt.gamma * minty_tl, ...
            - prod3(Ups_inv_K_tl_l' + opt.gamma * ups_tl', ...
                    prod3(solve_chol3(R_tl, Dtheta_K_tl), inv_K_tl_tl)) ...
            + prod3(inv_K_l_l', ...
                    prod3(tr(Dtheta_Ups_tl_l), inv_K_tl_tl)) ...
            + opt.gamma * prod3(tr(Dtheta_ups_tl), inv_K_tl_tl) ...
                                );
        
    % Now perform the correction to our variance
    var_ev = var_ev + sum(reshape(int_ml_Dtheta_mtl.^2, D, 1 , 1) .* V_theta);
end

% sanity check
if var_ev < 0
    warning('variance of evidence negative');
    fprintf('variance of evidence: %g\n', var_ev.*exp(samples.max_log_l)^2);
    var_ev = eps;
end
% var_ev has been determined using the rescaled log-likelihoods (that have
% had the maximum log likelihood subtracted off), we return correct values
% by scaling back again)
log_var_evidence = 2*samples.max_log_l + log(var_ev);

% Compute the second moment of the evidence
% ======================================================

% second moment of the evidence
mean_second_moment =  mean_ev.^2 + var_ev;

% mean_second_moment has been determined using the rescaled log-likelihoods
% (that have had the maximum log likelihood subtracted off), we return
% correct values by scaling back again)
log_mean_second_moment = 2*samples.max_log_l + log(mean_second_moment);

% Store a lot of stuff in the ev_params structure for use by
% expected_uncertainty_evidence.m
% ======================================================

% Many of our quantities have different names in
% expected_uncertainty_evidence.m, we rename appropriately
ev_params = struct(...
  'candidate_locations' , x_c, ...
  'sqd_dist_stack_s' , sqd_dist_stack_s, ...
  'R_l_s', R_l, ...
  'K_l_s', K_l, ...
  'R_tl_s' , R_tl, ...
  'K_tl_s' , K_tl, ...
  'inv_K_tl_s' , inv_K_tl_tl, ...
  'jitters_tl_s' , jitters_tl, ...
  'R_del_sc' , R_del, ...
  'K_del_sc' , K_del, ...
  'ups_l_s' , ups_l, ...
  'ups_del_sc' , ups_del, ...
  'Ups_sc_s' , Ups_del_l, ...
  'del_inv_K' , del_inv_K_del, ...
  'delta_tl_sc' , delta_tl_sc, ...
  'minty_del' , minty_del, ...
  'log_mean_second_moment', log_mean_second_moment ...
   );
if opt.sds_tl_log_input_scales
    ev_params.Dtheta_K_tl_s = Dtheta_K_tl;
end
