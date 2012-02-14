function [mean_log_evidence, var_log_evidence, sample_locs, sample_vals] = ...
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
[ais_mean_log_evidence, ais_var_log_evidence, sample_locs, sample_vals] = ...
    ais_mh(loglik_fn, prior, opt);

[sample_locs, sample_vals] = ...
    remove_duplicate_samples(sample_locs, sample_vals);
opt.num_samples = length(sample_vals);

% Now call BMC using the exp of those samples.
[mean_evidence, var_evidence] = ...
    bmc_integrate(sample_locs, exp(sample_vals - max(sample_vals)), prior);

% Todo: move these into the conversion eqns for better numerical stability.
% Also, possibly move the scaling into bmc_integrate.
mean_evidence = mean_evidence * exp( max(sample_vals));
var_evidence = var_evidence * exp(2*max(sample_vals));

% Convert the distribution over the evidence into a distribution over the
% log-evidence by moment matching.
var_log_evidence = log( var_evidence / mean_evidence^2 + 1 );
mean_log_evidence = log(mean_evidence);% - 0.5*var_log_evidence;

if var_log_evidence < 0
    warning('variance of log evidence negative');
    fprintf('variance of log evidence: %g\n', var_log_evidence);
    var_log_evidence = eps;
end
end
