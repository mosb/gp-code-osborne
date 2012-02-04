function [log_ev, log_var_ev, all_sample_locations, r_gp] = ...
    sbq(log_r_fn, prior_struct, opt)
% Take samples samples_mat so as to best estimate the
% evidence, an integral over exp(log_r_fn) against the prior in prior_struct.
% 
% OUTPUTS
% - samples_mat: m*n matrix of hyperparameter samples
% - log_ev: our mean estimate for the log of the evidence
% - log_var_ev: the variance for the log of the evidence
% - r_gp: takes fields:
% 
% INPUTS
% - start_pt: 1*n vector expressing starting point for algorithm
% - log_r_fn: a function that takes a single argument, a 1*n vector of
%             hyperparameters, and returns the log of the likelihood
% - prior_struct: requires fields
%                 * means
%                 * sds
% - opt: takes fields:
%        * num_samples: the number of samples to draw. If opt is a number rather
%          than a structure, it's assumed opt = num_samples.
%        * print: print reassuring dots.
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
%                              parameters.  Can be: one of:
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

sample_dimension = numel(prior_struct.mean);

% Set unspecified fields to default values.
default_opt = struct('num_samples', 300, ...
                     'num_retrains', 5, ...
                     'train_gp_time', 60, ...
                     'parallel', true, ...
                     'train_gp_num_samples', 10, ...
                     'train_gp_print', false, ...
                     'exp_loss_evals', 50 * sample_dimension, ...
                     'start_pt', zeros(1, sample_dimension), ...
                     'print', true, ...
                     'plots', false, ...
                     'set_ls_var_method', 'laplace');%'lengthscale');
names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end

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

% Define the box with which to bound the selection of samples.
lower_bound = prior_struct.mean - 5*sqrt(diag(prior_struct.covariance))';
upper_bound = prior_struct.mean + 5*sqrt(diag(prior_struct.covariance))';
direct_opts.maxevals = opt.exp_loss_evals;
direct_opts.showits = 0;
bounds = [lower_bound; upper_bound]';


% Start of actual SBQ algorithm
% =======================================

all_sample_locations = nan(opt.num_samples, sample_dimension);
all_sample_values = nan(opt.num_samples, 1);


next_sample_point = opt.start_pt;
for i = 1:opt.num_samples

    % Update sample structs.
    % ==================================
    all_sample_locations(i,:) = next_sample_point;          % Record the current sample location.
    all_sample_values(i,:) = log_r_fn(next_sample_point);   % Sample the integrand at the new point.

    % Grab all existing function samples and put them in a struct.
    samples = struct();
    samples.samples = all_sample_locations(1:i, :);
    samples.log_r = all_sample_values(1:i,:);
    samples.scaled_r = exp(samples.log_r - max(samples.log_r));
    
    
    % Retrain GP
    % ===========================   
    retrain_now = i >= retrain_inds(1);  % If we've passed the next retraining index.
    if i==1  % First iteration.

        % Set up GP without training it, because there's not enough data.
        gp_train_opt.optim_time = 0;
        [r_gp, quad_r_gp] = train_gp('sqdexp', 'constant', [], ...
                                     samples.samples, samples.scaled_r, gp_train_opt);
        gp_train_opt.optim_time = opt.train_gp_time;
        
    elseif retrain_now
        % Retrain gp.
        [r_gp, quad_r_gp] = train_gp('sqdexp', 'constant', r_gp, ...
                                     samples.samples, samples.scaled_r, gp_train_opt);
        retrain_inds(1) = [];   % Move to next retraining index. 
    else
        % for hypersamples that haven't been moved, update
        r_gp = revise_gp(samples.samples, samples.scaled_r, ...
                         r_gp, 'update', i);
    end
    
    % Put the values of the best quadrature parameters into the current GP.
    % NB: disp_hyperparams exponentiates the actual hyperparameters e.g.
    % the log-input-scales. 
    [best_hypersample, best_hypersample_struct] = disp_hyperparams(r_gp);
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
                quad_r_gp.quad_input_scales(r_gp.input_scale_inds);
        elseif strcmp(opt.set_ls_var_method, 'laplace')
            % Diagonal covariance whose diagonal elements set by using the laplace
            % method around the maximum likelihood value.

            % Assume the mean is the same as the mode.
            log_in_scale_means = log(r_gp_params.quad_input_scales);

            % Specify the likelihood function which we'll be taking the hessian of:
            like_func = @(log_in_scale) exp(log_gp_lik2( samples.samples, ...
                samples.scaled_r, ...
                r_gp, ...
                log(r_gp_params.quad_noise_sd), ...
                log_in_scale, ...
                log(r_gp_params.quad_output_scale), ...
                r_gp_params.quad_mean));

            % Find the Hessian
            laplace_sds = Inf;
            try
                laplace_sds = sqrt(-1./hessdiag( like_func, log_in_scale_means));
            catch e; 
                e;
            end

            % A little sanity checking, since at first the length scale won't be
            % sensible.
            bad_sd_ixs = isinf(laplace_sds) | (imag(laplace_sds) > 0);
            if any(bad_sd_ixs)
                warning('Infinite or positive lengthscales, Setting lengthscale variance to prior variance');
                good_sds = sqrt(diag(prior_struct.covariance));
                laplace_sds(bad_sd_ixs) = good_sds(bad_sd_ixs);
            end

            opt.sds_tr_input_scales = laplace_sds;

            % Plot the log-likelihood surface.
            if opt.plots && sample_dimension == 1
                figure(11); clf;
                hrange = linspace(-5, 10, 1000 );
                for t = 1:length(hrange)
                    vals(t) = like_func(hrange(t));
                end
                plot(hrange, vals, 'b'); hold on;
                y=get(gca,'ylim');
                h=plot([log_in_scale_means log_in_scale_means],y, 'g');

                % Plot the laplace-approx Gaussian.
                rescale = like_func(log_in_scale_means)/mvnpdf(0, 0, laplace_sds^2);
                plot(hrange, rescale.*mvnpdf(hrange', log_in_scale_means, laplace_sds^2), 'r'); hold on;
                xlabel('log input scale');
                ylabel('likelihood');
                legend(h, {'current mean of log input scale'})            
            end
        end
    end
    
    [log_ev, log_var_ev, r_gp_params] = log_evidence(samples, prior_struct, r_gp_params, opt);

    
    % Choose the next sample point.
    % =================================
    if i < opt.num_samples  % Except for the last iteration,
        
        % Define the criterion to be optimized.
        Problem.f = @(hs_a) expected_uncertainty_evidence...
                (hs_a', samples, prior_struct, r_gp_params, opt);
            
        if opt.plots && sample_dimension == 1

            % If we have a 1-dimensional function, optimize it by exhaustive
            % evaluation.
            N = 1000;
            test_pts = linspace(lower_bound, upper_bound, N);
            losses = nan(1, N);
            for loss_i=1:length(test_pts)
                losses(loss_i) = Problem.f(test_pts(loss_i));
            end
            % Choose the best point.
            [min_loss,min_ind] = min(losses);
            next_sample_point = test_pts(min_ind);
            
            % Plot the expected uncertainty surface.
            figure(1234); clf;
            h_surface = plot(test_pts, losses, 'b'); hold on;           
            % Also plot previously chosen points.
            test_pts = samples.samples;
            losses = nan(1, i);
            for loss_i=1:length(test_pts)
                losses(loss_i) = Problem.f(test_pts(loss_i));
            end
            h_points = plot(test_pts, losses, 'kd', 'MarkerSize', 4); hold on;
            h_best = plot(next_sample_point, min_loss, 'rd', 'MarkerSize', 4); hold on;
            xlabel('Sample location');
            ylabel('Expected uncertainty after adding a new sample');
            legend( [h_surface, h_points, h_best], {'Expected uncertainty', 'Previously Sampled Points', 'Best new sample'}, 'Location', 'Best');
            title(sprintf('Iteration %d', i ));
            drawnow;
        else
            % Call DIRECT to hopefully optimize faster than by exhaustive search.
            [exp_loss_min, next_sample_point] = Direct(Problem, bounds, direct_opts);
            next_sample_point = next_sample_point';
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
        fprintf('log evidence: %g +- %g i\n', log_ev, log_var_ev);
    end
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
