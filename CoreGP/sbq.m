function [mean_log_evidences, var_log_evidences, samples, gp_hypers] = ...
    sbq(log_likelihood_fn, prior, opt)
% Take samples samples_mat so as to best estimate the
% evidence, an integral over exp(log_r_fn) against the prior in prior_struct.
% 
% OUTPUTS
% - mean_log_evidences: our mean estimates for the log of the evidence
% - var_log_evidences: the variances for the log of the evidence
% - samples: n*d matrix of hyperparameter samples
% - gp_hypers
% 
% INPUTS
% - start_pt: 1*n vector expressing starting point for algorithm
% - log_likelihood_fn: a function that takes a single argument, a 1*n vector of
%                      hyperparameters, and returns the log of the likelihood.
% - prior: requires fields
%                 * means
%                 * sds
% - opt: takes fields:
%        * num_samples: the number of samples to draw. If opt is a number rather
%          than a structure, it's assumed opt = num_samples.
%        * print: If print == 1,  print reassuring dots. If print ==2,
%          print even more diagnostic information.
%        * num_retrains: how many times to retrain the gp throughout the
%          procedure. These are logarithmically spaced: early retrains are
%          more useful. 
%        * parallel: whether to use the parallel computing toolbox to make
%          training more efficient.
%        * train_gp_time: the amount of time in seconds to spend
%          (re)training the gp each time. The longer allowed, the more
%          local exploitation around each hyperparameter sample.
%        * train_gp_num_samples: how many hyperparameter samples to use for
%          the multi-start gradient descent procedure used for training.
%          The more, the more exploration.
%        * plots: Whether to plot the expected variance surface (only works in 1d)
%        * set_ls_var_method:  How to estimate the variance of lengthscale
%                              parameters.  Can be one of:
%            + 'lengthscale':  Use the squared lengthscale from quadrature params.
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
                     'num_retrains', 5, ...
                     'num_box_scales', 5, ...
                     'train_gp_time', 50 * D, ...
                     'parallel', true, ...
                     'train_gp_num_samples', 5 * D, ...
                     'train_gp_print', false, ...
                     'exp_loss_evals', 50 * D, ...
                     'start_pt', zeros(1, D), ...
                     'print', true, ...
                     'plots', false, ...
                     'set_ls_var_method', 'laplace');%'lengthscale');
opt = set_defaults( opt, default_opt );

% GP training options.
gp_train_opt.optim_time = opt.train_gp_time;
gp_train_opt.noiseless = true;
gp_train_opt.prior_mean = 0;
% print to screen diagnostic information about gp training
gp_train_opt.print = opt.train_gp_print;
% plot diagnostic information about gp training
gp_train_opt.plots = false;
gp_train_opt.parallel = opt.parallel;
gp_train_opt.num_hypersamples = opt.train_gp_num_samples;


% Specify iterations when we will retrain the GP on r.
retrain_inds = intlogspace(ceil(opt.num_samples/10), ...
                                opt.num_samples, ...
                                opt.num_retrains+1);
retrain_inds(end) = inf;



% Start of actual SBQ algorithm
% =======================================

next_sample_point = opt.start_pt;
for i = 1:opt.num_samples

    % Update sample struct.
    % ==================================
    samples.locations(i,:) = next_sample_point;          % Record the current sample location.
    samples.log_r(i,:) = log_likelihood_fn(next_sample_point);   % Sample the integrand at the new point.
    samples.scaled_r = exp(samples.log_r - max(samples.log_r));
    
    
    % Retrain GP
    % ===========================   
    retrain_now = i >= retrain_inds(1);  % If we've passed the next retraining index.
    if i==1  % First iteration.

        % Set up GP without training it, because there's not enough data.
        gp_train_opt.optim_time = 0;
        [gp_hypers, quad_r_gp] = train_gp('sqdexp', 'constant', [], ...
                                     samples.locations, samples.scaled_r, gp_train_opt);
        gp_train_opt.optim_time = opt.train_gp_time;
        
    elseif retrain_now
        % Retrain gp.
        [gp_hypers, quad_r_gp] = train_gp('sqdexp', 'constant', gp_hypers, ...
                                     samples.locations, samples.scaled_r, gp_train_opt);
        retrain_inds(1) = [];   % Move to next retraining index. 
    else
        % for hypersamples that haven't been moved, update
        gp_hypers = revise_gp(samples.locations, samples.scaled_r, ...
                         gp_hypers, 'update', i);
    end
    
    % Put the values of the best quadrature parameters into the current GP.
    % NB: disp_hyperparams exponentiates the actual hyperparameters e.g.
    % the log-input-scales. 
    [best_hypersample, best_hypersample_struct] = disp_hyperparams(gp_hypers);
    r_gp_params.quad_output_scale = best_hypersample_struct.output_scale;
    r_gp_params.quad_input_scales = best_hypersample_struct.input_scales;
    r_gp_params.quad_noise_sd = best_hypersample_struct.noise_sd;
    r_gp_params.quad_mean = 0;    

    
    % Update our posterior uncertainty about the lengthscale hyperparameters.
    % =========================================================================
    % Only if we've just updated the quadrature hyperparams.
    if i == 1 || retrain_now
        % We firstly assume that the input scales over the transformed r
        % surface are the same as for the r surface. We are secondly going to
        % assume that the posterior for the log-input scales over the
        % transformed r surface is a Gaussian. We take its mean to be the ML
        % value.  Its covariance is set immediately below:
        if strcmp(opt.set_ls_var_method, 'lengthscale')
            % Diagonal covariance whose diagonal elements are equal
            % to the squared input scales of a GP (quad_r_gp) fitted to the
            % likelihood of the transformed r surface as a function of log-input
            % scales.
            opt.sds_tr_input_scales = ...
                quad_r_gp.quad_input_scales(gp_hypers.input_scale_inds);
        elseif strcmp(opt.set_ls_var_method, 'laplace')
            % Diagonal covariance whose diagonal elements set by using the laplace
            % method around the maximum likelihood value.

            % Assume the mean is the same as the mode.
            laplace_mode = log(r_gp_params.quad_input_scales);

            % Specify the likelihood function which we'll be taking the hessian of:
            like_func = @(log_in_scale) exp(log_gp_lik2( samples.locations, ...
                samples.scaled_r, ...
                gp_hypers, ...
                log(r_gp_params.quad_noise_sd), ...
                log_in_scale, ...
                log(r_gp_params.quad_output_scale), ...
                r_gp_params.quad_mean));

            % Find the Hessian
            laplace_sds = Inf;
            try
                laplace_sds = sqrt(-1./hessdiag( like_func, laplace_mode));
            catch e; 
                e;
            end

            % A little sanity checking, since at first the length scale won't be
            % sensible.
            bad_sd_ixs = isnan(laplace_sds) ...
                | isinf(laplace_sds) ...
                | (abs(imag(laplace_sds)) > 0)...
                | laplace_sds > 20;
            if any(bad_sd_ixs)
                warning('Infinite or positive lengthscales, Setting lengthscale variance to prior variance.');
                good_sds = sqrt(diag(prior.covariance));
                laplace_sds(bad_sd_ixs) = good_sds(bad_sd_ixs);
            end
            opt.sds_tr_input_scales = laplace_sds;

            if opt.plots && D == 1
                plot_hessian_approx( like_func, laplace_sds, laplace_mode );
            end
        end
    end
    
    [log_ev, log_var_ev, r_gp_params] = log_evidence(samples, prior, r_gp_params, opt);

    
    % Choose the next sample point.
    % =================================
    if i < opt.num_samples  % Except for the last iteration,
        
        % Define the criterion to be optimized.
        objective_fn = @(new_sample_location) expected_uncertainty_evidence...
                (new_sample_location', samples, prior, r_gp_params, opt);
            
        % Define the box with which to bound the selection of samples.
        lower_bound = prior.mean - opt.num_box_scales*sqrt(diag(prior.covariance))';
        upper_bound = prior.mean + opt.num_box_scales*sqrt(diag(prior.covariance))';
        bounds = [lower_bound; upper_bound]';            
            
        if opt.plots && D == 1    
            % If we have a 1-dimensional function, optimize it by exhaustive
            % evaluation.
            [exp_loss_min, next_sample_point] = ...
                plot_1d_minimize(objective_fn, bounds, samples, log_var_ev);
        else
            % Do a local search around each of the candidate points, which
            % are, by design, far removed from existing evaluations.
            [exp_loss_min, next_sample_point] = ...
                min_around_points(objective_fn, r_gp_params.candidate_locations, ...
                3 * r_gp_params.quad_input_scales, opt.exp_loss_evals);
        end
    end
    
    % Print progress dots.
    if opt.print == 1
        if rem(i, 50) == 0
            fprintf('\n%g',i);
        else
            fprintf('.');
        end
    elseif opt.print == 2
        fprintf('evidence: %g +- %g\n', exp(log_ev), sqrt(exp(log_var_ev)));
    end
    
    % Convert the distribution over the evidence into a distribution over the
    % log-evidence by moment matching.  This is just a hack for now!!!
    mean_log_evidences(i) = log_ev;
    var_log_evidences(i) = log( exp(log_var_ev) / exp(log_ev)^2 + 1 );
end
end

function log_l = log_gp_lik2(X_data, y_data, gp, log_noise, ...
                             log_in_scale, log_out_scale, mean)
    % Evaluates the log_likelihood of a hyperparameter sample at a certain
    % set of hyperparameters.
    gp.grad_hyperparams = false;
    
    % Todo:  Make a function to package a hyperparameter sample into an array,
    % like the opposite of disp_hyperparams.  Use it to replace this next part,
    % and part of train_gp.m
    sample(gp.meanPos) = mean;
    sample(gp.noise_ind) = log_noise;
    sample(gp.input_scale_inds) = log_in_scale;
    sample(gp.output_scale_ind) = log_out_scale;
    gp.hypersamples(1).hyperparameters = sample;
    
    gp = revise_gp(X_data, y_data, gp, 'overwrite', [], 1);
    log_l = gp.hypersamples(1).logL;
end
