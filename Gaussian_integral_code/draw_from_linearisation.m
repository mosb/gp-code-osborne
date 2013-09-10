clear




% define observed likelihoodd lik & predictants
% ====================================

const = 10^(5*(rand-.5))

lik = ([0.1;0.25;0.1;0.15;0.18;0.15;0.48;0.05])/const;
n = length(lik);
x = linspace(0,10,n)';

xst = linspace(min(x)-10, max(x)+10, 1000)';

% define transform
% ====================================

mn = min(lik);
mx = max(lik);

beta = 0.99;
alpha = 0.01;

% Maximum likelihood approach to finding transform

ft = @(loglik,c) mn * exp(c(1) * loglik) + c(2) * loglik;
fdt = @(loglik,c) c(1) * mn * exp(c(1) * loglik) + c(2);

obj = @(c) ...
    (ft(0,c) - mn).^2 + ...
    (c(3) * fdt(0,c) - alpha .* (mx - mn)).^2 + ...
    (ft(c(3),c) - mx).^2 + ...
    (c(3) * fdt(c(3),c) - beta .* (mx - mn)).^2;

options = optimset('MaxFunEvals',4000);
c = fminunc(obj, ones(1,3),options);

t = @(loglik) ft(loglik,c);
clf
logliks = linspace(0,1,1000);
plot(logliks,t(logliks));

dt = @(loglik) fdt(loglik,c);
invt = @(lik) lik/c(2) ...
    - 1/c(1) * lambertw(mn * c(1)/c(2) * exp(c(1)/c(2) * lik));
    
% a = mn;
% b = 2 * (mx - mn) - beta/2;
% c = -(mx - mn) + beta/2;
% 
% 
% t = @(loglik) a + b * loglik.^2 + c * loglik.^4;
% dt = @(loglik) 2 * b * loglik + 4 * c * loglik.^3;

clf
logliks = linspace(0,c(3),1000);
plot(logliks,t(logliks));

% define gp over inv-transformed (e.g. log-) likelihoods
% ====================================

invt_lik = invt(lik);

% gp covariance hypers
w = 0.5;
h = std(invt_lik);
sigma = eps;
mu = min(invt_lik);

% define covariance function
fK = @(x, xd) h^2 * exp(-0.5*(x - xd).^2 / (w^2));
K = @(x, xd) bsxfun(fK, x, xd');

% Gram matrix
V = K(x, x) + eye(n)*sigma^2;

% GP posterior for loglik
m_loglik = mu + K(xst, x) * (V\(invt_lik-mu));
C_loglik = K(xst, xst) - K(xst, x) * (V\K(x, xst));
sd_loglik = sqrt(diag(C_loglik));

figure(1)
clf
gp_plot(xst, m_loglik, sd_loglik, x, invt(lik));
ylabel 'log-likelihood'

% define linearisation, lik = a * loglik + c
% ====================================

% best_loglik = max(invt(lik));

a = dt(m_loglik);
c = t(m_loglik) - a .* m_loglik;

sd_loglik(1)
a(1)

%a = exp(m_loglik) * exp(-mu/2);
% a = exp(mu/2)*(1 + (m_loglik-mu) + 1/2 * (m_loglik-mu).^2 ...
%      + 1/6 * (m_loglik-mu).^3 + 1/24 * (m_loglik-mu).^4);
% % c = a.*(1-m_loglik)
% c = a.* exp(mu/2) - a.*m_loglik;

% gp over likelihood
% ====================================

m_lik = diag(a) * m_loglik + c;
C_lik =  diag(a) * C_loglik * diag(a);
sd_lik = diag(C_lik);

figure(2)
clf
gp_plot(xst, m_lik, sd_lik, x, lik);
ylabel 'likelihood'
%plot(xst, a,'g')
