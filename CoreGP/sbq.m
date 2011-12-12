function [samples_mat, log_ev, r_gp] = ...
    sbq(start_pt, log_r_fn, prior_struct, opt)
% take hyperparameter samples samples_mat so as to best estimate the
% evidence, an integral over hyperparameter space of exp(log_r_fn).
% 
% OUTPUTS
% - samples_mat: m*n matrix of hyperparameter samples
% - log_evidence: the log of the evidence
% - r_gp: takes fields:
% 
% INPUTS
% - start_pt: 1*n vector expressing starting point for algorithm
% - log_r_fn: a function that takes a single argument, a 1*n vector of
% hyperparameters, and returns the log of the likelihood
% - prior_struct: requires fields
% * means
% * sds
% - opt: takes fields:
% num_samples: the number of samples to draw. If opt is a number rather
% than a structure, it's assumed opt = num_samples.
% print: print reassuring dots.

if nargin<5
    opt = struct();
elseif ~istruct(opt)
    num_samples = opt;
    opt = struct();
    opt.num_samples = num_samples;
end

hs = start_pt;
num_hps = size(hs,2);

default_opt = struct('num_samples', 300, ...
                    'num_retrains', 5, ...
                    'train_gp_time', 20, ...
                    'train_gp_num_samples', 10, ...
                    'exp_loss_evals', 50 * num_hps, ...
                    'print', true);
                
names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end

% train_opt defines the options for the training of the gp
% this will be temporarily overwritten below, but this is its usual value
train_opt.optim_time = opt.train_gp_time;
%train_opt.active_hp_inds = 2:8;
train_opt.prior_mean = 0;
train_opt.num_hypersamples = opt.train_gp_num_samples;

retrain_inds = intlogspace(ceil(opt.num_c/10), ...
                                    opt.num_samples, ...
                                    opt.num_retrains+1);
retrain_inds(end) = inf;

% define the box with which to bound the selection of samples
lower_bound = prior_struct.means - 5*prior_struct.sds;
upper_bound = prior_struct.means + 5*prior_struct.sds;

direct_opts.maxevals = opt.exp_loss_evals;
direct_opts.showits = 0;
bounds = [lower_bound; upper_bound]';

r_gp = set_gp('sqdexp', opt.mean_fn, [], X_data, y_data, ...
        opt.num_hypersamples);

% - sample_struct requires fields
% * samples
% * log_r
samples_mat = nan(opt.num_samples, num_hps);
log_r_mat = nan(opt.num_samples, 1);

for i = 1:opt.num_samples
    
    samples_mat(i,:) = hs;
    log_r_mat(i,:) = log_r_fn(hs);
    
    samples_mat_i = samples_mat(1:i, :);
    log_r_mat_i = log_r_mat(1:i,:);
    [max_log_r_mat_i, max_r_ind] = max(log_r_mat_i);
    log_r_mat_i = log_r_mat_i - max_log_r_mat_i;
        
    sample_struct = struct();
    sample_struct.samples = samples_mat_i;
    sample_struct.log_r = log_r_mat_i;

    retrain_now = i >= retrain_inds(1); 
    
    if i==1 
        
        train_opt.optim_time = 0;
        r_gp = train_gp('sqdexp', 'constant', [], ...
            sample_struct.samples, sample_struct.log_r, train_opt);
        train_opt.optim_time = opt.train_gp_time;
        
    elseif retrain_now
        
        % retrain gp
        
        r_gp = train_gp('sqdexp', 'constant', r_gp, ...
            sample_struct.samples, sample_struct.log_r, train_opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(r_gp);

        r_gp.quad_output_scale = best_hypersample_struct.output_scale;
        r_gp.quad_input_scales = best_hypersample_struct.input_scales;
        r_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        r_gp.quad_mean = 0;
        
        retrain_inds(1) = [];
        
        % WARBNING: if this is actually necessary, better have a look at
        % spgpgo and gpgo; I don't understand why if overwriting is
        % necessary, it's not done immediately after retraining
        r_gp = revise_gp(sample_struct.samples, sample_struct.log_r, ...
            r_gp, 'overwrite', []);

    else
        % for hypersamples that haven't been moved, update
        r_gp = revise_gp(sample_struct.samples, sample_struct.log_r, ...
            r_gp, 'update', i);
    end
    
    [log_ev, r_gp] = ...
    log_evidence(sample_struct, prior_struct, r_gp, opt);

    if i < opt.num_samples
        % minimise the expected uncertainty to find the next sample
            
        Problem.f = @(hs_a) log_expected_uncertainty_evidence...
                (hs_a', sample_struct, prior_struct, r_gp, opt);
        [exp_loss_min, hs] = Direct(Problem, bounds, direct_opts);
        hs = hs';
        
        
    end
    
    if opt.print
        fprintf('.');
    end
    
end