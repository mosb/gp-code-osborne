function [mean_log_evidence, var_log_evidence, samples, sample_vals, stats] = ...
    ais_mh(loglik_fn, prior, opt)
% Annealed Importance Sampling w.r.t. a Gaussian prior
% using a Metropolis-Hastings sampler with a Gaussian proposal distribution.
%
% Runs Mentropolis-Hastings over tempered versions of the posterior
% so as to best estimate the evidence, an integral over input space
% of p(x)exp(log_r_fn(x)).
%
% More info at:
% http://www.cs.toronto.edu/~radford/ftp/ais-rev.pdf
%
% Inputs:
%   loglik_fn: a function that takes a single argument, a 1*d vector, and
%              returns the log of the likelihood.
%   prior: requires fields
%          * mean
%          * covariance
%   opt: takes fields:
%        * num_samples: the number of samples to draw.
%        * proposal_covariance
%        * learn_proposal: if true, calls ais_mh to tune the acceptance ratio
%                          to approximately
% 
% Outputs:
%   mean_log_evidence: the mean of our poterior over the log of the evidence.
%   var_log_evidence: the variance of our posterior over the log of the
%                     evidence.
%   samples: n*d matrix of the locations of the samples.
%   sample_vals:
%   stats: a struct of stats about the run, containing:
%          * weights: n*1 list of weights.
%          * acceptance ratio.
%
%
% David Duvenaud
% January 2012


% Define default options.
if nargin < 3
    opt.num_samples = 1000;
end

% Todo: set this adaptively with a burn-in?
opt.proposal_covariance = prior.covariance./10;

% Define annealing schedule.  This can be anything, as long as it starts
% from zero and doesn't go above one.
temps = linspace( 0, 1, opt.num_samples + 1);

% Define annealed pdf.
log_prior_fn = @(x) logmvnpdf(x, prior.mean, prior.covariance);
log_annealed_pdf = @(x, temp) temp.*loglik_fn(x) + log_prior_fn(x);

% Allocate memory.
weights = nan(opt.num_samples, 1);
samples = nan(opt.num_samples, numel(prior.mean));

% Start with a sample from the prior.
cur_pt = mvnrnd( prior.mean, prior.covariance );

num_accepts = 1;
for t = 2:length(temps)
    
    % Compute MH proposal.
    proposal = mvnrnd( cur_pt, opt.proposal_covariance );
    proposal_ll = log_annealed_pdf(proposal, temps(t));
    cur_pt_ll = log_annealed_pdf(cur_pt, temps(t));
    
    % Possibly take a MH step.
    ratio = exp(proposal_ll - cur_pt_ll);
    if ratio > rand
        num_accepts = num_accepts + 1;
        cur_pt = proposal;
    end

    % Compute weights.
    sample_vals(t) = loglik_fn(cur_pt);
    weights(t) = sample_vals(t) * (temps(t) - temps(t - 1));
    samples(t, :) = cur_pt;
end

stats.acceptance_ratio = num_accepts / length(temps);
fprintf('\nAcceptance ratio: %f \n', stats.acceptance_ratio);

weights(1) = [];
samples(1, :) = [];
sample_vals(1) = [];
mean_log_evidence = sum(weights);

%rho = auto_correlation(weights);
%effective_sample_size = opt.num_samples * ( 1 - rho ) / (1 + rho );
var_log_evidence = NaN;%var(weights)*opt.num_samples/effective_sample_size^2;  % todo: double check this.
end


