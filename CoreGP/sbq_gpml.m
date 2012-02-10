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
default_opt = struct('num_samples', 300, ...
                     'exp_loss_evals', 50 * D, ...
                     'start_pt', zeros(1, D), ...
                     'init_pts', 10 * D, ...  % Number of points to start with.
                     'plots', false, ...
                     'set_ls_var_method', 'laplace');
opt = set_defaults( opt, default_opt );


% Initialize with some random points.
for i = 1:opt.init_pts
    next_sample_point = mvnrnd(prior.mean, prior.covariance);
    samples.locations(i,:) = next_sample_point;
    samples.log_r(i,:) = log_likelihood_fn(next_sample_point);
end

% Start of actual SBQ algorithm
% =======================================
next_sample_point = opt.start_pt;
for i = opt.init_pts + 1:opt.num_samples

    % Update sample struct.
    % ==================================
    samples.locations(i,:) = next_sample_point;          % Record the current sample location.
    samples.log_r(i,:) = log_likelihood_fn(next_sample_point);   % Sample the integrand at the new point.
    samples.scaled_r = exp(samples.log_r - max(samples.log_r));
    
    
    % Train a GP over the untransformed observations.
    % ====================================================
    inference = @infExact;
    likfunc = @likGauss;
    meanfunc = {'meanZero'};
    max_iters = 100;    
    covfunc = @covSEiso;
    
    % Init GP Hypers each time at first to prevent getting lost in some weird place.
    if ~exist('gp_hypers', 'var') || i < 10 * D
        fprintf('Initializing hypers from heuristics');
        gp_hypers.mean = [];%0;
        gp_hypers.lik = log(0.01);
        %gp_hypers.cov = log( [ 1 1] );%log([ones(1, D) 1]);    
        gp_hypers.cov = log( [ mean(sqrt(diag(prior.covariance)))/2 1] ); 
    end
  
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
            %[grad,err,finaldelta] = gradest(fun,x0)
        catch e; 
            e;
        end

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
        fprintf('Posterior variance in lengthscales: '); disp(laplace_sds);
    end
    
    % Convert gp_hypers to r_gp_params.
    % TODO: check that these are the right units.
    fprintf('Output variance: '); disp(exp(gp_hypers.cov(end)));
    fprintf('Lengthscales: '); disp(exp(gp_hypers.cov(1:end - 1)));    
    r_gp_params.quad_output_scale = exp(gp_hypers.cov(end));
    r_gp_params.quad_input_scales(1:D) = exp(gp_hypers.cov(1:end - 1));
    [log_ev, log_var_ev, r_gp_params] = log_evidence(samples, prior, r_gp_params, opt);
    
    % Choose the next sample point.
    % =================================
    if i < opt.num_samples  % Except for the last iteration,
        
        % Define the criterion to be optimized.
        objective_fn = @(new_sample_location) expected_uncertainty_evidence...
                (new_sample_location', samples, prior, r_gp_params, opt);
            
        % Define the box with which to bound the selection of samples.
        lower_bound = prior.mean - 5*sqrt(diag(prior.covariance))';
        upper_bound = prior.mean + 5*sqrt(diag(prior.covariance))';
        bounds = [lower_bound; upper_bound]';            
            
        if opt.plots && D == 1    
            % If we have a 1-dimensional function, optimize it by exhaustive
            % evaluation.
            [exp_loss_min, next_sample_point] = ...
                plot_1d_minimize(objective_fn, bounds, samples, log_var_ev);
        else
            % Call DIRECT to hopefully optimize faster than by exhaustive search.
            problem.f = objective_fn;
            direct_opts.maxevals = opt.exp_loss_evals;
            direct_opts.showits = 1;
            [exp_loss_min, next_sample_point] = Direct(problem, bounds, direct_opts);
            next_sample_point = next_sample_point';
        end
    end
    
    % Print progress.
    fprintf('Iteration %d evidence: %g +- %g\n', i, exp(log_ev), sqrt(exp(log_var_ev)));

    % Convert the distribution over the evidence into a distribution over the
    % log-evidence by moment matching.  This is just a hack for now!!!
    mean_log_evidences(i) = log_ev;
    var_log_evidences(i) = log( exp(log_var_ev) / exp(log_ev)^2 + 1 );
end
end


function l = gpml_likelihood( log_in_scale, gp_hypers, inference, meanfunc, covfunc, likfunc, X, y)
% Just replaces the lengthscales.
    gp_hypers.cov(1:end-1) = log_in_scale;
    l = exp(-gp_fixedlik(gp_hypers, inference, meanfunc, covfunc, likfunc, X, y));
end


function l = gpml_likelihood_grad( log_in_scale, gp_hypers, inference, meanfunc, covfunc, likfunc, X, y)
% Just replaces the lengthscales.
    gp_hypers.cov(1:end-1) = log_in_scale;
    l = exp(-gp_fixedlik(gp_hypers, inference, meanfunc, covfunc, likfunc, X, y));
end

