function [mean_log_evidences, var_log_evidences, samples, gp_hypers] = ...
    sbq_gpml_ais(log_likelihood_fn, prior, opt)
% Take samples samples_mat so as to best estimate the
% evidence, an integral over exp(log_r_fn) against the prior in prior_struct.
% 
% This version uses GPML to set hyperparams, and AIS to choose points.
%
% OUTPUTS
% - mean_log_evidences: our mean estimate for the log of the evidence
% - var_log_evidences: the variance for the log of the evidence
% - sample_locations: n*d matrix of hyperparameter samples
% - gp_hypers
% 
% INPUTS
% - log_likelihood_fn: a function that takes a single argument, a 1*n vector of
%                      hyperparameters, and returns the log of the likelihood.
% - prior: requires fields
%                 * means
%                 * sds
% - opt: takes fields:
%        * num_samples: the number of samples to draw. If opt is a number rather
%          than a structure, it's assumed opt = num_samples.
%        * plots: Whether to plot the expected variance surface (only works in 1d)
%        * set_ls_var_method:  How to estimate the variance of lengthscale
%                              parameters.  Can be one of:
%            + 'laplace': Compute the Hessian of the log-likelihood surface.
%            + 'none': Assume zero variance in the lengthscales.


% Initialize options.
% ===========================
if nargin<3
    opt = struct();
elseif ~isstruct(opt)
    num_samples = opt;
    opt = struct();
    opt.num_samples = num_samples;
end

D = numel(prior.mean);

% Set unspecified fields to default values.
default_opt = struct('num_samples', 300, ...
                     'exp_loss_evals', 50 * D, ...
                     'start_pt', zeros(1, D), ...
                     'plots', false, ...
                     'set_ls_var_method', 'laplace');
opt = set_defaults( opt, default_opt );


% Get sample locations from a run of AIS.
[ais_mean_log_evidence, ais_var_log_evidence, sample_locs, sample_vals] = ...
    ais_mh(log_likelihood_fn, prior, opt);


% Update sample struct.
% ==================================
samples.locations = sample_locs;
for i = 1:opt.num_samples
    % Sample the integrand at the new point.
    samples.log_r(i,:) = log_likelihood_fn(samples.locations(i,:));
end
samples.scaled_r = exp(samples.log_r - max(samples.log_r));
    
    
% Train GP
% ===========================   
inference = @infExact;
likfunc = @likGauss;
meanfunc = {'meanZero'};
max_iters = 100;

% Init GP Hypers each time to prevent getting lost in some weird place.
covfunc = @covSEiso;
gp_hypers.mean = [];
gp_hypers.lik = log(0.01);  % function values go between 0 and 1.
%gp_hypers.cov = log( [ 1 1] );%log([ones(1, D) 1]);    
gp_hypers.cov = log( [ mean(sqrt(diag(prior.covariance)))/2 1] ); 


% Fit the model, but not the likelihood hyperparam (which stays fixed).
gp_hypers = minimize(gp_hypers, @gp_fixedlik, -max_iters, ...
                     inference, meanfunc, covfunc, likfunc, ...
                     samples.locations, samples.scaled_r);     
    
% Update our posterior uncertainty about the lengthscale hyperparameters.
% =========================================================================
if strcmp(opt.set_ls_var_method, 'laplace')
    % Set the variances of the lengthscales using the laplace
    % method around the maximum likelihood value.
    laplace_mode = gp_hypers.cov(1:end - 1);

    % Specify the likelihood function which we'll be taking the hessian of:
    % Todo: check that there isn't a scaling factor since Mike uses
    %       Gaussians, and Carl uses sqexp.
    %       Also, there is the matter of scaling the values.
    % Todo: Compute the Hessian using the gradient instead.
    like_func = @(log_in_scale) gpml_likelihood( log_in_scale, gp_hypers, ...
        inference, meanfunc, covfunc, likfunc, samples.locations, ...
        samples.scaled_r);

    % Find the Hessian.
    laplace_sds = Inf;
    try
        laplace_sds = sqrt(-1./hessdiag( like_func, laplace_mode));
    catch e; e; end

    % A little sanity checking, since at first the length scale won't be
    % sensible.
    bad_sd_ixs = isnan(laplace_sds) | isinf(laplace_sds) | (abs(imag(laplace_sds)) > 0);
    if any(bad_sd_ixs)
        warning(['Infinite or positive lengthscales, ' ...
                'Setting lengthscale variance to prior variance']);
        good_sds = sqrt(diag(prior.covariance));
        laplace_sds(bad_sd_ixs) = good_sds(bad_sd_ixs);
    end
    opt.sds_tr_input_scales = laplace_sds;

    if opt.plots && D == 1
        plot_hessian_approx( like_func, laplace_sds, laplace_mode );
    end
end
    
laplace_sds
exp(gp_hypers.cov(end))
exp(gp_hypers.cov(1:end - 1))

% Convert gp_hypers to r_gp_params.  GPML and Mike's code have different 
% normalization constants.
log_conversion_constant = ...
    -logmvnpdf(zeros(1,D), zeros(1,D), diag(ones(D,1).*exp(gp_hypers.cov(1:end - 1))))/2;
converted_output_scale = gp_hypers.cov(end) + log_conversion_constant;
opt.sds_tr_input_scales = opt.sds_tr_input_scales * exp(log_conversion_constant);fprintf('Output variance: '); disp(exp(converted_output_scale));
fprintf('Lengthscales: '); disp(exp(gp_hypers.cov(1:end - 1)));    
r_gp_params.quad_output_scale = exp(converted_output_scale);
r_gp_params.quad_input_scales(1:D) = exp(gp_hypers.cov(1:end - 1));
[log_ev, log_var_ev, r_gp_params] = log_evidence(samples, prior, r_gp_params, opt);

% Convert the distribution over the evidence into a distribution over the
% log-evidence by moment matching.  This is just a hack for now!!!
mean_log_evidences(1) = log_ev;
var_log_evidences(1) = log( exp(log_var_ev) / exp(log_ev)^2 + 1 );
end

function l = gpml_likelihood( log_in_scale, gp_hypers, inference, meanfunc, covfunc, likfunc, X, y)
    % Just replaces the lengthscales.
    gp_hypers.cov(1:end-1) = log_in_scale;
    l = exp(-gp_fixedlik(gp_hypers, inference, meanfunc, covfunc, likfunc, X, y));
end
