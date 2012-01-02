
r_mean = 1;
r_sd = 1;

log_r_fn = @(x) log(normpdf(x,r_mean,r_sd));

start_pt = -3;
prior_struct.means = 0;
prior_struct.sds = 0.5;

opt.num_retrains = 3;
opt.num_samples = 100;

[samples_mat, log_ev, r_gp] = ...
    sbq(start_pt, log_r_fn, prior_struct, opt);

clf
test_pts = linspace(prior_struct.means - 5*prior_struct.sds, ...
            prior_struct.means + 5*prior_struct.sds, 1000);
plot(test_pts, log_r_fn(test_pts), 'b')
hold on
plot(samples_mat, log_r_fn(samples_mat), '.k', 'MarkerSize', 8)
xlim([-5 5])

exact = ...
    lognormpdf(r_mean, prior_struct.means, sqrt(prior_struct.sds^2 + r_sd^2))
log_ev