function y = log_lognpdf(z_p,z_mu,z_sigma_squared)
% y = log_lognpdf(z_p,z_mu,z_sigma_squared)
% approximate the distribution over a log-transformed variable z
% p(z) = N(z; z_mu, z_sigma_squared)
% so as to give a Gaussian over the untransformed variable
% p(x) = N(x; x_mu, x_sigma_squared)
% and evaluate the log-density at exp(z_p).

if nargin<1
    error(message('stats:lognpdf:TooFewInputs'));
end
if nargin < 2
    z_mu = 0;
end
if nargin < 3
    z_sigma_squared = 1;
end

x_p = exp(z_p);

% Return NaN for out of range parameters.
z_sigma_squared(z_sigma_squared <= 0) = NaN;

x_mu = exp(z_mu + 0.5 * z_sigma_squared);
x_sigma_squared = exp(2 * z_mu) * ...
            (exp(2 * z_sigma_squared) - exp(z_sigma_squared));

y = lognpdf(x_p, x_mu, x_sigma_squared);