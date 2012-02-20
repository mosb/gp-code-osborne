function [log_mean_evidences, log_var_evidences, samples] = ...
    online_bmc(loglik_fn, prior, opt)
% Makes a fixed-length sampler into an online sampler, the slow way.
% Simply re-fixes the random seed and calls the sampler again.
%
% David Duvenaud
% February 2012

[log_mean_evidences, log_var_evidences, samples] = ...
    make_online_slow(@bmc, loglik_fn, prior, opt);
end
