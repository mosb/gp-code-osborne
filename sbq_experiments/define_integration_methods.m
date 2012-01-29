function methods = define_integration_methods()
% Define all the methods with which to solve an integral.
%
% These methods should all have the signature:
%
% [ mean_log_evidence, var_log_evidence, samples ] = ...
%   integrate( log_likelihood_fn, prior )
%
% Where
%   -  loglik_fn is a function that takes a single argument, a 1*d vector, and
%      returns the log of the likelihood.
%   -  prior requires fields
%             * mean
%             * covariance

% Specify integration methods.
methods = {};
methods{end+1} = @simple_monte_carlo;
%methods{end+1} = @importance_sampling;
methods{end+1} = @ais_mh;
%methods{end+1} = @sbq;
