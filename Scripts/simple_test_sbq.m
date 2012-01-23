
r_mean1 = 1;
r_sd1 = 1;

r_mean2 = 4;
r_sd2 = 1;

log_r_fn = @(x) log(normpdf(x,r_mean1,r_sd1)+normpdf(x,r_mean2,r_sd2));

start_pt = -3;
prior_struct.means = 0;
prior_struct.sds = 1;

opt.num_retrains = 5;
opt.num_samples = 50;
opt.plots = true;

[samples_mat, log_ev, r_gp] = ...
    sbq(start_pt, log_r_fn, prior_struct, opt);

clf
test_pts = linspace(prior_struct.means - 5*prior_struct.sds, ...
            prior_struct.means + 5*prior_struct.sds, 1000);
plot(test_pts, log_r_fn(test_pts), 'b')
hold on
plot(samples_mat, log_r_fn(samples_mat), '.k', 'MarkerSize', 8)
xlim([-5 5])

% the exact log-evidence
exact = ...
log(normpdf(r_mean1, prior_struct.means, sqrt(prior_struct.sds^2 + r_sd1^2))...
+normpdf(r_mean2, prior_struct.means, sqrt(prior_struct.sds^2 + r_sd2^2)))
% estimated log evidence
log_ev