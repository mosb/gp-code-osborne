function [log_mean_evidence, log_var_evidence, samples] = ...
    simple_monte_carlo(loglik_fn, prior, opt)
% Simple Monte Carlo
% 
% Inputs:
% - loglik_fn: a function that takes a single argument, a 1*d vector,
%             and returns the log of the likelihood
% - prior: requires fields
%          * mean
%          * covariance
% - opt: takes fields:
%        * num_samples: the number of samples to draw.
%
%
% Outputs:
%   mean_log_evidence: the mean of our poterior over the log of the evidence.
%   var_log_evidence: the variance of our posterior over the log of the
%                     evidence.
% - samples : n*d matrix of samples
%
%
%
% David Duvenaud
% January 2012

if nargin < 3
    opt.num_samples = 1000;
end

% Draw samples.
samples = mvnrnd( prior.mean, prior.covariance, opt.num_samples );
logliks = loglik_fn( samples );

% Remove any bad likelihoods
good_ix = ~isinf(logliks);
num_good = sum(good_ix);

% Compute empirical mean.
log_mean_evidence = logsumexp(logliks(good_ix)) - log(num_good);

% Compute empirical variance. todo:  check if this is the right thing to do.
log_var_evidence = logsumexp(2.*(logliks(good_ix) - log_mean_evidence)) - log(num_good);
end
