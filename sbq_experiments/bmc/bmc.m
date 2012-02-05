function [mean_log_evidence, var_log_evidence, samples, sample_vals] = ...
    bmc(loglik_fn, prior, opt)
% Naive Bayesian Monte Carlo.  Chooses samples based on AIS.
%
% Based on: http://mlg.eng.cam.ac.uk/zoubin/papers/RasGha03.pdf
%
% Inputs:
%   loglik_fn: a function that takes a single argument, a 1*d vector, and
%              returns the log of the likelihood.
%   prior: requires fields
%          * mean
%          * covariance
%   opt: takes fields:
%        * num_samples: the number of samples to draw.
%        * proposal_covariance: for the AIS.
% 
% Outputs:
%   mean_log_evidence: the mean of our poterior over the log of the evidence.
%   var_log_evidence: the variance of our posterior over the log of the
%                     evidence.
%   samples: n*d matrix of the locations of the samples.
%   weights: n*1 list of weights.
%
%
% David Duvenaud
% January 2012


% Define default options.
if nargin < 3
    opt.num_samples = 100;
end

% Get sample locations from a run of AIS.
[ais_mean_log_evidence, ais_var_log_evidence, samples, sample_vals] = ...
    ais_mh(loglik_fn, prior, opt);

% Now call BMC using the exp of those samples.
[mean_evidence, var_evidence] = ...
    bmc_integrate(samples, exp(sample_vals), prior);

% Convert the distribution over the evidence into a distribution over the
% log-evidence by moment matching.
var_log_evidence = log( var_evidence / mean_evidence^2 + 1 );
mean_log_evidence = log(mean_evidence);% - 0.5*var_log_evidence;
end
