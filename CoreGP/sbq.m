function [log_mean_evidences, log_var_evidences, samples, ...
    l_gp_hypers, tl_gp_hypers, del_gp_hypers, ev_params] = ...
    sbq(log_likelihood_fn, prior, opt)
% Take samples samples_mat so as to best estimate the
% evidence, an integral over exp(log_l_fn) against the prior in prior_struct.
% 
% OUTPUTS
% - mean_log_evidences: our mean estimates for the log of the evidence;
%       the ith element corresponds to our mean after i samples
% - var_log_evidences: the variances for the log of the evidence;
%       the ith element corresponds to our variance after i samples
% - samples: n*d matrix of hyperparameter samples - tl_gp_hypers: the
% (quadrature) hyperparameters of the gp fitted to the log-likelihood
% surface
% 
% INPUTS
% - start_pt: 1*n vector expressing starting point for algorithm
% - log_likelihood_fn: a function that takes a single argument, a 1*n vector of
%                      hyperparameters, and returns the log of the likelihood.
% - prior: requires fields
%                 * mean
%                 * covariance
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
%        * marginalise_scales: Whether to approximately marginalise the log
%          input scales of the gp over the log-likelihood


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
                     'gamma', 1, ...
                     'num_retrains', 10, ...
                     'num_box_scales', 5, ...
                     'train_gp_time', 50 * D, ...
                     'parallel', true, ...
                     'train_gp_num_samples', 5 * D, ...
                     'train_gp_print', false, ...
                     'exp_loss_evals', 500 * D^2, ...
                     'start_pt', prior.mean, ...
                     'print', true, ...
                     'plots', false, ...
                     'marginalise_scales', true);%'lengthscale');
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
    samples.log_l(i,:) = log_likelihood_fn(next_sample_point);   % Sample the integrand at the new point.
    samples.max_log_l = max(samples.log_l); % all log-likelihoods have max_log_l subtracted off
    samples.scaled_l = exp(samples.log_l - samples.max_log_l);
    samples.tl = log_transform(samples.scaled_l, opt.gamma);
    
    % Retrain GP
    % ===========================   
    retrain_now = i >= retrain_inds(1);  % If we've passed the next retraining index.
    if i==1  % First iteration.

        % Set up GP without training it, because there's not enough data.
        gp_train_opt.optim_time = 0;
        [l_gp, quad_l_gp] = train_gp('sqdexp', 'constant', [], ...
                                     samples.locations, samples.scaled_l, ...
                                     gp_train_opt);
                                 
        [tl_gp, quad_tl_gp] = train_gp('sqdexp', 'constant', [], ...
                                     samples.locations, samples.tl, ...
                                     gp_train_opt);
                                 
                                 
        gp_train_opt.optim_time = opt.train_gp_time;
        
        del_gp_hypers = [];
        
    elseif retrain_now
        % Retrain gp.
        [l_gp, quad_l_gp] = train_gp('sqdexp', 'constant', l_gp, ...
                                     samples.locations, samples.scaled_l, ...
                                     gp_train_opt);
                                 
        [tl_gp, quad_tl_gp] = train_gp('sqdexp', 'constant', tl_gp, ...
                                     samples.locations, samples.tl, ...
                                     gp_train_opt);            
                                 
        del_gp_hypers = [];
                                 
        retrain_inds(1) = [];   % Move to next retraining index. 
    else
        % for hypersamples that haven't been moved, update
        l_gp = revise_gp(samples.locations, samples.scaled_l, ...
                         l_gp, 'update', i);
                     
        tl_gp = revise_gp(samples.locations, samples.tl, ...
                         tl_gp, 'update', i);
    end
    
    % Put the values of the best hyperparameters into dedicated structures.
    l_gp_hypers = best_hyperparams(l_gp);
    tl_gp_hypers = best_hyperparams(tl_gp);
    % hyperparameters for gp over delta, the difference between log-gp-mean-r and
    % gp-mean-log-r
    % del_gp_hypers = del_hyperparams(tl_gp_hypers);
    
    [log_mean_evidences(i), log_var_evidences(i), ev_params, del_gp_hypers] = ...
        log_evidence(samples, prior, l_gp_hypers, tl_gp_hypers, del_gp_hypers, opt);

    tic;
    % Choose the next sample point.
    % =================================
    if i < opt.num_samples  % Except for the last iteration,
        
        % Define the criterion to be optimized.
        objective_fn = @(new_sample_location) expected_uncertainty_evidence...
                (new_sample_location(:)', samples, prior, ...
                l_gp_hypers, tl_gp_hypers, del_gp_hypers, ev_params, opt);
            
        % Define the box with which to bound the selection of samples.
        lower_bound = prior.mean - ...
            opt.num_box_scales*sqrt(diag(prior.covariance))';
        upper_bound = prior.mean + ...
            opt.num_box_scales*sqrt(diag(prior.covariance))';
        bounds = [lower_bound; upper_bound]';            
            
        if opt.plots && D == 1    
            % If we have a 1-dimensional function, optimize it by exhaustive
            % evaluation.
            [exp_loss_min, next_sample_point] = ...
                plot_1d_minimize(objective_fn, bounds, samples, log_var_evidences(i));
        else
            
            % Search within the prior box.
            [exp_loss_min, next_sample_point] = ...
                min_in_box( objective_fn, prior, ...
                samples, tl_gp_hypers, opt.exp_loss_evals );
        end
    end
    toc
    
    % Print progress dots.
    if opt.print == 1
        if rem(i, 50) == 0
            fprintf('\n%g',i);
        else
            fprintf('.');
        end
    elseif opt.print == 2
        fprintf('evidence: %g +- %g\n', exp(log_mean_evidences(i)), ...
                                        sqrt(exp(log_var_evidences(i))));
    end
end
end
