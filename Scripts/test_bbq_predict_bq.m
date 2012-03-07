clear;
%cd ~/Code/GP/BQR
% 


problem_bbq_predict_bq;

opt.num_samples = 30;
[log_ev, log_var_ev, samples, ...
    l_gp_hypers_SE, tl_gp_hypers_SE, del_gp_hypers_SE, ev_params] = ...
    sbq(log_l_fn, prior, opt);

samples.qd = q_fn(samples.locations);

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



qd_gp = train_gp('sqdexp', 'constant', [], ...
                             samples.locations, samples.qd, ...
                             gp_train_opt);

% Put the values of the best hyperparameters into dedicated structures.
qd_gp_hypers_SE = best_hyperparams(qd_gp);

[mean_out, sd_out, unadj_mean_out, unadj_sd_out] = ...
    predict_bq(sample_struct, prior, ...
    l_gp_hypers_SE, tl_gp_hypers_SE, del_gp_hypers_SE, ...
    qd_gp_hypers_SE, qdd_gp_hypers_SE, opt);