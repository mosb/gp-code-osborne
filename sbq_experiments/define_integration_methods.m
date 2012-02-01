function methods = define_integration_methods()
% Define all the methods with which to solve an integral.
%
% These methods should all have the signature:
%
% [ mean_log_evidence, var_log_evidence, samples ] = ...
%   integrate( log_likelihood_fn, prior, opt )
%
% Where
%   - loglik_fn is a function that takes a single argument, a 1*d vector, and
%     returns the log of the likelihood.
%   - prior requires fields
%            * mean
%            * covariance
%   - opt is a struct for holding different options.
%
%
%
%
%

smc_method.nicename = 'Simple Monte Carlo';
smc_method.uniquename = 'simple monte carlo v1';
smc_method.acronym = 'SMC';
smc_method.function = @simple_monte_carlo;
smc_method.opt = [];

ais_method.nicename = 'Annealed Importance Sampling';
ais_method.uniquename = 'annealed importance sampling v1';
ais_method.acronym = 'AIS';
ais_method.function = @ais_mh;
ais_method.opt = [];

sbq_method.nicename = 'Sequential Bayesian Quadrature';
sbq_method.uniquename = 'sequential bayesian quadrature v1';
sbq_method.acronym = 'SBQ';
sbq_method.function = @sbq;
sbq_method.opt = [];

bmc_method.nicename = 'Vanilla Bayesian Monte Carlo';
bmc_method.uniquename = 'vanilla bayesian monte carlo v1';
bmc_method.acronym = 'BMC';
bmc_method.function = @bmc;
bmc_method.opt = [];


% Specify integration methods.
methods = {};
methods{end+1} = smc_method;
methods{end+1} = ais_method;
methods{end+1} = bmc_method;
methods{end+1} = sbq_method;
