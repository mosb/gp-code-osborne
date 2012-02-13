function [mean_log_evidences, var_log_evidences, samples] = ...
    online_ais_mh(loglik_fn, prior, opt)
% Makes a fixed-length sampler into an online sampler, the slow way.
% Simply re-fixes the random seed and calls the sampler again.
%
% David Duvenaud
% February 2012


[mean_log_evidences, var_log_evidences, samples] = ...
    make_online_slow(@ais_mh, loglik_fn, prior, opt);

% Now, find the empirical predictive variance.
if ~isfield(opt, 'num_variance_estimation_runs')
    opt.num_variance_estimation_runs = 10;
end

for i = 1:opt.num_variance_estimation_runs
    all_mean_log_evidences(i,:) = make_online_slow(@ais_mh, loglik_fn, prior, opt);
end

var_log_evidences = var(all_mean_log_evidences);

end
