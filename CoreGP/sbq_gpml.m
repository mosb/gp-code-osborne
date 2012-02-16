function [mean_log_evidences, var_log_evidences, samples, gp_hypers] = ...
    sbq_gpml(log_likelihood_fn, prior, opt)
% Take samples samples_mat so as to best estimate the
% evidence, an integral over exp(log_r_fn) against the prior in prior_struct.
% 
% This version uses GPML to set hyperparams.
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
%        * start_pt: 1*n vector expressing starting point for algorithm
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
default_opt = struct('num_samples', 100, ...
                     'exp_loss_evals', 150 * D, ...
                     'start_pt', prior.mean, ...
                     'num_prior_pts', 10, ...  % Start with samples from prior.
                     'gamma', 1, ...
                     'plots', false, ...
                     'set_ls_var_method', 'laplace');
opt = set_defaults( opt, default_opt );


% Initialize with some random points.
for i = 1:opt.num_prior_pts
    next_sample_point = mvnrnd(prior.mean, prior.covariance);
    samples.locations(i,:) = next_sample_point;
    samples.log_l(i,:) = log_likelihood_fn(next_sample_point);
end

% Start of actual SBQ algorithm
% =======================================
next_sample_point = opt.start_pt;
for i = opt.num_prior_pts + 1:opt.num_samples

    % Update sample struct.
    % ==================================
    samples.locations(i,:) = next_sample_point;          % Record the current sample location.
    samples.log_l(i,:) = log_likelihood_fn(next_sample_point);   % Sample the integrand at the new point.
    samples.max_log_l = max(samples.log_l); % all log-likelihoods have max_log_l subtracted off
    samples.scaled_l = exp(samples.log_l - samples.max_log_l);
    samples.tl = log_transform(samples.scaled_l, opt.gamma);

    % Train GPs
    % ===========================   
    inference = @infExact;
    likfunc = @likGauss;
    meanfunc = {'meanZero'};
    max_iters = 1000;
    covfunc = @covSEiso;

    % Init GP Hypers.
    init_hypers.mean = [];
    init_hypers.lik = log(0.01);  % Values go between 0 and 1, so no need to scale.
    init_lengthscales = mean(sqrt(diag(prior.covariance)))/10;
    init_output_variance = .1;
    init_hypers.cov = log( [init_lengthscales init_output_variance] ); 

    % Fit the model, but not the likelihood hyperparam (which stays fixed).
    fprintf('Fitting GP to observations...\n');
    gp_hypers = init_hypers;
    gp_hypers = minimize(gp_hypers, @gp_fixedlik, -max_iters, ...
                         inference, meanfunc, covfunc, likfunc, ...
                         samples.locations, samples.scaled_l);
    if any(isnan(gp_hypers.cov))
        gp_hypers = init_hypers;
        warning('Optimizing hypers failed');
    end
    l_gp_hypers.log_output_scale = gp_hypers.cov(end);
    l_gp_hypers.log_input_scales(1:D) = gp_hypers.cov(1:end - 1);
    fprintf('Output variance: '); disp(exp(l_gp_hypers.log_output_scale));
    fprintf('Lengthscales: '); disp(exp(l_gp_hypers.log_input_scales));

    fprintf('Fitting GP to log-observations...\n');
    gp_hypers_log = init_hypers;
    gp_hypers_log = minimize(gp_hypers_log, @gp_fixedlik, -max_iters, ...
                             inference, meanfunc, covfunc, likfunc, ...
                             samples.locations, samples.tl);        
    if any(isnan(gp_hypers_log.cov))
        gp_hypers_log = init_hypers;
        warning('Optimizing hypers on log failed');
    end
    tl_gp_hypers.log_output_scale = gp_hypers_log.cov(end);
    tl_gp_hypers.log_input_scales(1:D) = gp_hypers_log.cov(1:end - 1);
    fprintf('Output variance of logL: '); disp(exp(tl_gp_hypers.log_output_scale));
    fprintf('Lengthscales on logL: '); disp(exp(tl_gp_hypers.log_input_scales));

    if opt.plots
        figure(50); clf;
        gpml_plot( gp_hypers, samples.locations, samples.scaled_l);
        title('GP on untransformed values');
        figure(51); clf;
        gpml_plot( gp_hypers_log, samples.locations, samples.tl);
        title('GP on log( exp(scaled) + 1) values');
    end

    % Optionally set uncertainty in lengthscales using the laplace
    % method around the maximum likelihood value.
    if strcmp(opt.set_ls_var_method, 'laplace')
        % Specify the likelihood function which we'll be taking the hessian of:
        like_func = @(log_in_scale) gpml_lengthscale_likelihood( log_in_scale, ...
            gp_hypers_log, inference, meanfunc, covfunc, likfunc, ...
            samples.locations, samples.tl);

        laplace_mode = gp_hypers_log.cov(1:end - 1);
        failsafe_sds = sqrt(diag(prior.covariance));
        opt.sds_tl_log_input_scales = ...
            likelihood_laplace( like_func, laplace_mode, failsafe_sds);

        if opt.plots && D == 1
            plot_hessian_approx( like_func, opt.sds_tl_log_input_scales, laplace_mode );
        end
    end

    [log_mean_evidence, log_var_evidence, ev_params, del_gp_hypers] = ...
        log_evidence(samples, prior, l_gp_hypers, tl_gp_hypers, [], opt);

    % Choose the next sample point.
    % =================================
    if i < opt.num_samples  % Except for the last iteration.

        % Define the criterion to be optimized.
        objective_fn = @(new_sample_location) expected_uncertainty_evidence...
                (new_sample_location(:)', samples, prior, ...
                l_gp_hypers, tl_gp_hypers, del_gp_hypers, ev_params, opt);
            
        % Define the box with which to bound the selection of samples.
        lower_bound = prior.mean - 5*sqrt(diag(prior.covariance))';
        upper_bound = prior.mean + 5*sqrt(diag(prior.covariance))';
        bounds = [lower_bound; upper_bound]';            
            
        if opt.plots && D == 1    
            % If we have a 1-dimensional function, optimize it by exhaustive
            % evaluation.
            [exp_loss_min, next_sample_point] = ...
                plot_1d_minimize(objective_fn, bounds, samples, log_var_evidence);
        else
            % Search within the prior box.
            [exp_loss_min, next_sample_point] = ...
                min_in_box( objective_fn, prior, opt.exp_loss_evals );
        end
    end
    
    % Print progress.
    fprintf('Iteration %d evidence: %g +- %g\n', i, ...
            exp(log_mean_evidence), sqrt(exp(log_var_evidence)));

    % Convert the distribution over the evidence into a distribution over the
    % log-evidence by moment matching.  This is just a hack for now!!!
    mean_log_evidences(i) = log_mean_evidence;
    var_log_evidences(i) = log( exp(log_var_evidence) / exp(log_mean_evidence)^2 + 1 );
end
end
