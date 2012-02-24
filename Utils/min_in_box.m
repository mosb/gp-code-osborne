function [exp_loss_min, next_sample_point] = ...
    min_in_box( objective_fn, prior, samples, tl_gp_hypers, evals )
%
% Searches within a prior box.

num_starts = 10;
evals_per_start = ceil(evals/num_starts);

optim_opts = ...
    optimset('GradObj','off',...
    'Display','off', ...
    'MaxFunEvals', evals_per_start,...
    'LargeScale', 'off',...
    'Algorithm','interior-point'...
    );


lower_bound = prior.mean - 3.*sqrt(diag(prior.covariance))';
upper_bound = prior.mean + 3.*sqrt(diag(prior.covariance))';

[max_log_l, max_ind] = max(samples.log_l);
max_sample = samples.locations(max_ind, :);
scales = exp(tl_gp_hypers.log_input_scales);
exploit_starts = find_candidates(max_sample, 1, scales, 1);

explore_starts = mvnrnd(prior.mean, prior.covariance, num_starts-1);
    
starts = [exploit_starts;explore_starts];

for i = 1:num_starts
    cur_start_pt = starts(i, :);
    [next_sample_point, exp_loss_min] = ...
        fmincon(objective_fn,cur_start_pt, ...
        [],[],[],[],...
        lower_bound,upper_bound,[],...
        optim_opts);
end
end
