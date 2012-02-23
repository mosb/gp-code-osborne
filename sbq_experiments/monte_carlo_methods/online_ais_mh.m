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

%var_log_evidences = var(all_mean_log_evidences)' ./ opt.num_variance_estimation_runs;



for t = 1:opt.num_samples
    cur_mean_ev = logsumexp(all_mean_log_evidences(:,t)) - log(opt.num_variance_estimation_runs);
    var_log_evidences(t) = logsumexp(2.*(all_mean_log_evidences(:,t) - cur_mean_ev)) - log(num_good);
end


% Now convert to a distribution over Z instead of LogZ.
%[log_mean_evidences, log_var_evidences] = ...
%    log_of_normal_to_log_normal( mean_log_evidences, var_log_evidences );

end
