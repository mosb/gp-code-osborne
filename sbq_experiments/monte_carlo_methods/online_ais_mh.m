function [mean_log_evidences, var_log_evidences, samples] = ...
    online_ais_mh(loglik_fn, prior, opt)
% Makes a fixed-length sampler into an online sampler, the slow way.
% Simply re-fixes the random seed and calls the sampler again.
%
% David Duvenaud
% February 2012

[mean_log_evidences, var_log_evidences, samples] = ...
    make_online_slow(@ais_mh, loglik_fn, prior, opt);
end
