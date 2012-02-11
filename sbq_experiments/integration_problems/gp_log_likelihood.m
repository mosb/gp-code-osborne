function log_likelihood = gp_log_likelihood( log_hypers, X, y, covfunc )
% An attempted reconstruction of the integration problem tackled in the original
% BMC paper.
%
% Uses a SE ARD kernel, so has D + 2 hypers.
%
% David Duvenaud
% February 2012
% =====================

[N,D] = size(X);

prior_mean = zeros(D + 2, 1);
prior_var = 4.*ones(D + 2, 1);
prior_log_likelihood = logmvnpdf( log_hypers, prior_mean, prior_var );

inference = @infExact;
likfunc = @likGauss;
meanfunc = {'meanZero'};

gp_hypers.cov = log_hypers(1:end-1);
gp_hypers.lik = log_hypers(end);
data_log_likelihood = gp(gp_hypers, inference, meanfunc, covfunc, likfunc, X, y);

log_likelihood = prior_log_likelihood + data_log_likelihood;
end
