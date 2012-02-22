function [log_mean_evidences, log_var_evidences, samples, gp_hypers] = ...
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
                     'exp_loss_evals', 150 * D^2, ...
                     'start_pt', prior.mean, ...
                     'start_with_sds', true, ...  % Start with the prior +-1 1sd and .
                     'gamma', 1, ...
                     'plots', false, ...
                     'marginalise_scales', true);
opt = set_defaults( opt, default_opt );


% Initialize with some random points.
%for i = 1:opt.num_prior_pts
%    next_sample_point = mvnrnd(prior.mean, prior.covariance);
%    samples.locations(i,:) = next_sample_point;
%    samples.log_l(i,:) = log_likelihood_fn(next_sample_point);
%end

sample_points = [];
if opt.start_with_sds
    
    for d = 1:D
        sample_points = [sample_points; zeros(4,D)];
        sample_points(end - 3, d) = prior.mean(d) + sqrt(prior.covariance(d, d));
        sample_points(end - 2, d) = prior.mean(d) + 2 * sqrt(prior.covariance(d, d));
        sample_points(end - 1, d) = prior.mean(d) - sqrt(prior.covariance(d, d));
        sample_points(end, d) = prior.mean(d) - 2 * sqrt(prior.covariance(d, d));
    end
    samples.locations = sample_points;
    for i = 1:size(sample_points, 1);
        samples.log_l(i,:) = log_likelihood_fn(sample_points(i,:));   
    end
end

% Start of actual SBQ algorithm
% =======================================
next_sample_point = opt.start_pt;
if size(sample_points,1) >= opt.num_samples
    warning('sbq: no active sampling performed');
end

for i = size(sample_points,1) + 1:opt.num_samples

    % Update sample struct.
    % ==================================
    samples.locations(i,:) = next_sample_point;          % Record the current sample location.
    samples.log_l(i,:) = log_likelihood_fn(next_sample_point);   % Sample the integrand at the new point.
    samples.max_log_l = max(samples.log_l); % all log-likelihoods have max_log_l subtracted off
    samples.scaled_l = exp(samples.log_l - samples.max_log_l);
    samples.tl = log_transform(samples.scaled_l, opt.gamma);

    % Train GPs
    % ===========================   
    fprintf('Fitting GP to observations...\n');
    lengthscale_center = mean(sqrt(diag(prior.covariance))) / 10;
    gp_hypers = ...
        fit_hypers_multiple_restart( samples.locations, samples.scaled_l, ...
                                     lengthscale_center );
    l_gp_hypers.log_output_scale = gp_hypers.cov(end);
    l_gp_hypers.log_input_scales(1:D) = gp_hypers.cov(1:end - 1);
    fprintf('Output variance: '); disp(exp(l_gp_hypers.log_output_scale));
    fprintf('Lengthscales: '); disp(exp(l_gp_hypers.log_input_scales));

    fprintf('Fitting GP to log-observations...\n');
    lengthscale_center = mean(sqrt(diag(prior.covariance))) / 10;
    gp_hypers_log = ...
        fit_hypers_multiple_restart( samples.locations, samples.tl,...
                                     lengthscale_center );
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

    [log_mean_evidences(i), log_var_evidences(i), ev_params, del_gp_hypers] = ...
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
                plot_1d_minimize(objective_fn, bounds, samples, log_var_evidences(i));
        else
            % Search within the prior box.
            [exp_loss_min, next_sample_point] = ...
                min_in_box( objective_fn, prior, opt.exp_loss_evals );
        end
    end
    
    % Print progress.
    fprintf('Iteration %d log evidence: %g +- %g\n', i, ...
            log_mean_evidences(i), log_var_evidences(i));
end
end

function best_hypers = fit_hypers_multiple_restart( X, y, l )
    
    % Specify GP Model.
    inference = @infExact;
    likfunc = @likGauss;
    meanfunc = {'meanZero'};
    covfunc = @covSEiso;
    opt_min.length = -1000;
    opt_min.verbosity = 0;
    
    % Init GP Hypers.
    init_hypers.mean = [];
    init_hypers.lik = log(0.01);  % Values go between 0 and 1, so no need to scale.
    init_output_variance = .1;
    
    init_lengthscales = [l * 100, l * 10, l, l / 10, l / 100];
    
    for i = 1:length(init_lengthscales);
        init_hypers.cov = log( [init_lengthscales(i) init_output_variance] ); 
        % Fit the model, but not the likelihood hyperparam (which stays fixed).    
        [gp_hypers{i} fX] = minimize(init_hypers, @gp_fixedlik, opt_min, ...
                                     inference, meanfunc, covfunc, likfunc, ...
                                     X, y);
        nll(i) = fX(end);
    end
    [best_nll, best_ix] = min(nll);
    fprintf('NLL of different lengthscale mins: '); disp(nll);
    best_hypers = gp_hypers{best_ix};
    if any(isnan(best_hypers.cov))
        best_hypers = init_hypers;
        warning('Optimizing hypers failed');
    end    
end
