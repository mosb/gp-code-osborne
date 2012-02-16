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
smc_method.function = @online_smc;%@simple_monte_carlo;
smc_method.opt = [];

ais_method.nicename = 'Annealed Importance Sampling';
ais_method.uniquename = 'annealed importance sampling v1';
ais_method.acronym = 'AIS';
ais_method.function = @online_ais_mh; %@ais_mh;
ais_method.opt = [];

bmc_method.nicename = 'Vanilla Bayesian Monte Carlo';
bmc_method.uniquename = 'vanilla bayesian monte carlo ais v1';
bmc_method.acronym = 'BMC';
bmc_method.function = @online_bmc; %@bmc;
bmc_method.opt = [];

log_bmc_ais_method.nicename = 'Log Bayesian Monte Carlo';
log_bmc_ais_method.uniquename = 'log bmc ais v1';
log_bmc_ais_method.acronym = 'LBMC';
log_bmc_ais_method.function = @online_log_bmc;
log_bmc_ais_method.opt = [];

sbq_method.nicename = 'Doubly Bayesian Quadrature';
sbq_method.uniquename = 'sequential bayesian quadrature mike v1';
sbq_method.acronym = 'BBQ Mike';
sbq_method.function = @sbq;
sbq_method.opt = [];

sbq_gpml_method.nicename = 'Doubly Bayesian Quadrature GPML';
sbq_gpml_method.uniquename = 'sbq gpml v1';
sbq_gpml_method.acronym = 'BBQ';
sbq_gpml_method.function = @sbq_gpml;
sbq_gpml_method.opt = [];

bq_gpml_ais_method.nicename = 'Bayesian Quadrature using AIS';
bq_gpml_ais_method.uniquename = 'bayesian quadrature gpml ais v1';
bq_gpml_ais_method.acronym = 'BQ';
bq_gpml_ais_method.function = @online_bq_ais;
bq_gpml_ais_method.opt.set_ls_var_method = 'none';




% Specify integration methods.
methods = {};
methods{end+1} = smc_method;
methods{end+1} = ais_method;
methods{end+1} = bmc_method;
methods{end+1} = sbq_method;
methods{end+1} = sbq_gpml_method;
methods{end+1} = bq_gpml_ais_method;

%methods{end+1} = log_bmc_ais_method;
