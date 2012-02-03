
% Set up a simple toy function to test SQB.
r_mean1 = 1;
r_sd1 = 1;
r_mean2 = 4;
r_sd2 = 1;
normf = @(x,m,sd) (2*pi*sd^2)^(-0.5)*exp(-0.5*(x-m).^2/sd^2);
log_r_fn = @(x) log(normf(x,r_mean1,r_sd1)+normf(x,r_mean2,r_sd2));

% Set parameters of SBQ

prior_struct.mean = 0;
prior_struct.covariance = 1;
opt.print=2;
opt.num_retrains = 5;
opt.train_gp_time = 20;
opt.num_samples = 50;
opt.plots = true;
opt.parallel = false;
opt.set_ls_var_method = 'laplace'; %'laplace';
opt.start_pt = -3;

[log_ev, log_var_ev, samples_mat, r_gp] = ...
    sbq(log_r_fn, prior_struct, opt);

% Plot integrand and sample points.
test_pts = linspace(prior_struct.mean - 5*prior_struct.covariance, ...
            prior_struct.mean + 5*prior_struct.covariance, 1000);
figure;
h_func = plot(test_pts, log_r_fn(test_pts), 'b'); hold on;
h_samples = plot(samples_mat, log_r_fn(samples_mat), '.k', 'MarkerSize', 8)
xlim([-5 5]);
legend( [h_func, h_samples], {'Log-integrand', 'Sample Points'}, 'Location', 'Best');


% the exact log-evidence
exact = ...
log(normpdf(r_mean1, prior_struct.mean, sqrt(prior_struct.covariance + r_sd1^2))...
+normpdf(r_mean2, prior_struct.mean, sqrt(prior_struct.covariance + r_sd2^2)))
% estimated log evidence
log_ev
