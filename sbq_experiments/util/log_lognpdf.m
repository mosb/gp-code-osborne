function y = log_lognpdf(z_p,z_mu,z_sigma_squared)
% y = log_lognpdf(z_p,z_mu,z_sigma_squared)
% use the distribution over a log-transformed variable z
% p(z) = N(z; z_mu, z_sigma_squared)
% to evaluate the log-density of the untransformed variable x at exp(z_p).

if nargin<1
    error(message('stats:lognpdf:TooFewInputs'));
end
if nargin < 2
    z_mu = 0;
end
if nargin < 3
    z_sigma_squared = 1;
end

% Return NaN for out of range parameters.
z_sigma_squared(z_sigma_squared <= 0) = NaN;

try
    y = -0.5 * ((z_p - z_mu).^2)./sigma_squared ...
        - z_p - log(sqrt(2*pi)) - 0.5*log(z_sigma_squared);
catch
    error(message('stats:lognpdf:InputSizeMismatch'));
end