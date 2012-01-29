function log_Z = log_volume_between_two_gaussians( mu_a, sigma_a, mu_b, sigma_b)
% Find the volume of the integral of the product of two multivariate Gaussians.
%
% Inputs:
%   mu_a, sigma_a are the mean and covariance of the first gaussian.
%   mu_b, sigma_b are the mean and covariance of the second gaussian.
%
% David Duvenaud
% January 2012

a_inv = inv(sigma_a);
b_inv = inv(sigma_b);
sigma_c = inv(a_inv + b_inv);

log_Z = -0.5.*(mu_a - mu_b) * ((sigma_a \ sigma_c) / sigma_b) * (mu_a - mu_b)' ...
        - 0.5.*logdet( 2*pi .* (sigma_a * sigma_b) * (a_inv + b_inv) );
end

function ld = logdet(K)
    % returns the log-determinant of posdef matrix K
    ld = 2*sum(log(diag(chol(K))));
end
