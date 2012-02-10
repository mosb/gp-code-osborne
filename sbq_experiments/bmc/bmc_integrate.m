function [expected_Z, variance] = ...
    bmc_integrate(X, y, prior, covfunc, hypers_init, learn_hypers)
% Uses Bayesian Monte Carlo to compute the expected area under a function.
%
% Arguments:
%   X:             sample locations.
%   y:             function evaluations.
%   prior.mean:      prior mean.
%   prior.covariance:   prior variance.
%   covfunc:       the kernel function of the GP model used.
%   hypers_init:   the hyperparameters of the kernel function.
%   learn_hypers:  if true, optimize hyperparameters.
%
% Returns:
%   expected_Z:    the mean estimate of Z.
%   variance:      variance of Z.
%   
% David Duvenaud
% November 2011

D = numel(prior.mean);
N = length(y);

if nargin < 4; covfunc = @covSEiso; end
if nargin < 5
    hypers_init.mean = [];
    hypers_init.lik = log(0.01);
    %hypers_init.cov = log( [ 1 1] );%log([ones(1, D) 1]);
    hypers_init.cov = log( [ mean(sqrt(diag(prior.covariance)))/2 1] ); 
end
if nargin < 6; learn_hypers = true; end


% Optimize hyperparameters
if learn_hypers
    % Set up GP model.
    inference = @infExact;
    likfunc = @likGauss;
    meanfunc = {'meanZero'};
    max_iters = 100;

    % Fit the model, but not the likelihood hyperparam (which stays fixed).
    [hypers, nlZ] = minimize(hypers_init, @gp_fixedlik, -max_iters, ...
                        inference, meanfunc, covfunc, likfunc, X, y');                
    exp_hypers = [ exp(hypers.cov)]
else
    hypers = hypers_init;
end

% Fill in gram matrix
K = covfunc( hypers.cov, X ) + diag(ones(N,1)) .* exp(2*hypers.lik);

% Formulas from Carl and Zoubin's paper for the mean and variance.
w_lengths = exp(hypers.cov(1));
w_0 = exp(2*hypers.cov(2));
b = prior.mean';
B = prior.covariance';
a = X;
A = diag( ones( D, 1 ) .* w_lengths.^2 );
c = w_0 ./ sqrt(det(A\B + eye(D)));
 
for i = 1:N
    z(i) = c .* exp( -0.5 * ( a(i,:)' - b)' * ((A + B) \ (a(i,:)' - b)));
end

expected_Z = z * (K \ y');
variance = w_0 ./ sqrt(det(2.*(A \ B) + eye(D))) - z * (K \ z');
end

