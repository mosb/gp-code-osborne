function [exp_loss_min, next_sample_point] = ...
    min_in_box( objective_fn, prior, samples, tl_gp_hypers, evals )
%
% Searches within a prior box.

num_starts = 100;
evals_per_start = ceil(evals/num_starts);
num_exploits = ceil(num_starts/10);

optim_opts = ...
    optimset('GradObj','off',...
    'Display','off', ...
    'MaxFunEvals', evals_per_start,...
    'LargeScale', 'off',...
    'Algorithm','interior-point'...
    );

num_dims = length(prior.mean);

lower_bound = prior.mean - 3.*sqrt(diag(prior.covariance))';
upper_bound = prior.mean + 3.*sqrt(diag(prior.covariance))';

[max_log_l, max_ind] = max(samples.log_l);
max_sample = samples.locations(max_ind, :);
scales = exp(tl_gp_hypers.log_input_scales);
exploit_starts = find_candidates(max_sample, num_exploits, scales, 1);

continued_starts = ...
    find_candidates(samples.locations(end, :), num_exploits, scales, 1);

explore_starts = mvnrnd(prior.mean, prior.covariance, num_starts-2*num_exploits);
    
starts = [exploit_starts;continued_starts;explore_starts];

end_points = nan(num_starts,num_dims);
end_exp_loss = nan(num_starts,1);
for i = 1:num_starts
    cur_start_pt = starts(i, :);
    [end_points(i,:), end_exp_loss(i)] = ...
        fmincon(objective_fn,cur_start_pt, ...
        [],[],[],[],...
        lower_bound,upper_bound,[],...
        optim_opts);
end
[exp_loss_min, best_ind] = min(end_exp_loss);
next_sample_point = end_points(best_ind, :);

end
