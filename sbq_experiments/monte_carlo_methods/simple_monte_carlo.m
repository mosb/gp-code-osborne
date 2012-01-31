function [mean_log_evidence, var_log_evidence, samples] = ...
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

% Compute empirical mean.
mean_log_evidence = logsumexp(logliks) - log(opt.num_samples);

% Compute empirical variance. todo:  check if this is the right thing to do.
var_log_evidence = var(logliks) / opt.num_samples;
end
