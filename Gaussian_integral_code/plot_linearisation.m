function draw_from_linearisation

plotdir = '~/Docs/bayes-quadrature-for-gaussian-integrals-paper/figures/';

% beta controls how wide the error bars are for large likelihoods
beta = 10;
% alpha sets how wide the error bars are for small likelihoods
alpha = 1/100;

colours = cbrewer('qual','Paired', 12);
colours = colours(2:2:end, :);

% NB: results seem fairly insensitive to selection of alpha (sigh). beta
% has desired effect.

% define observed likelihood, lik, & locations of predictants
% ====================================

% multiply fixed likelihood function by this constant
const = 1;%10^(5*(rand-.5));

%lik = rand(6,1)/const;
lik = ([0.1;0.25;0.1;0.15;0.18;0.15;0.48;0.05])/const;
n = length(lik);
x = linspace(0,10,n)';

sub_xst = -5;
add_xst = 5;

n_st = 1000;
xst = linspace(min(x) + sub_xst, max(x) + add_xst, n_st)';

% define map f(tlik) = lik
% ====================================

mn = min(lik);
mx = max(lik);

% Maximum likelihood approach to finding map

f_h = @(tlik, theta) mn + theta(1) * tlik + theta(2) * tlik.^2;
df_h = @(tlik, theta) theta(1) + 2 * theta(2) * tlik;
ddf_h = @(tlik, theta) 2 * theta(2);
invf_h = @(lik, theta) 0.5 * 1/theta(2) * (...
            -theta(1) + sqrt(theta(1)^2 - 4 * theta(2) * (mn - lik) ));

options.MaxFunEvals = 1000;
[logth,min_obj] = fminunc(@(logth) objective(logth, f_h, df_h, mn, mx, ...
    alpha, beta), ...
    zeros(1,3), options);
min_obj
theta = exp(logth);

f = @(tlik) f_h(tlik,theta);
df = @(tlik) df_h(tlik, theta);
ddf = @(tlik) ddf_h(tlik, theta);
invf = @(lik) invf_h(lik, theta);

figure(3)
clf
n_tliks = 1000;
tliks = linspace(0, theta(3), n_tliks);
plot(tliks,f(tliks),'k');
hold on
%plot(tliks,df(tliks),'r');

xlabel 'transformed likelihood'
ylabel 'likelihood'
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(gcf, 'color', 'white'); 
set(gca, 'YGrid', 'off');
set(gca, 'color', 'white'); 

% define gp over inv-transformed (e.g. log-) likelihoods
% ====================================

invf_lik = invf(lik);

% gp covariance hypers
w = 1;
h = std(invf_lik);
sigma = eps;
mu = min(invf_lik);

% define covariance function
fK = @(x, xd) h^2 .* exp(-0.5*(x - xd).^2 / w^2);
K = @(x, xd) bsxfun(fK, x, xd');

% Gram matrix
V = K(x, x) + eye(n)*sigma^2;

% GP posterior for tlik
m_tlik = mu + K(xst, x) * (V\(invf_lik-mu));
C_tlik = K(xst, xst) - K(xst, x) * (V\K(x, xst));
sd_tlik = sqrt(diag(C_tlik));

figure(1)
clf
subplot(2, 1, 1)
params.legend_location = 0;
params.y_label = sprintf('transformed\n likelihood');
params.x_label = 0;
gp_plot(xst, m_tlik, sd_tlik, x, invf(lik), [], [], params);
axis tight;


% define linearisation, lik = a * tlik + c
% ====================================


% exact linearisation
a = df(m_tlik);
c = f(m_tlik) - a .* m_tlik;

% unnecessary under the quadratic transform
% % approximate linearisation
% best_tlik = mu;
% a = df(best_tlik) + (m_tlik - best_tlik) .* ddf(best_tlik);
% c = f(m_tlik) - a .* m_tlik;

% gp over likelihood
% ====================================

m_lik = diag(a) * m_tlik + c;
C_lik =  diag(a) * C_tlik * diag(a);
sd_lik = sqrt(diag(C_lik));

figure(1)
subplot(2, 1, 2)
params.legend_location = 0;
params.y_label = 'likelihood';
params.x_label = '$x$';
gp_plot(xst, m_lik, sd_lik, x, lik, [], [], params);
axis tight;
%plot(xst, a,'g')


% plot linearisation
% ====================================

ind = 0;
for i = round((-sub_xst + [-1, 5, 7.9])/(range(xst)) * n_st)
    ind = ind + 1;
    colour = colours(ind, :);
    
    figure(3)
    
    % plot linearisations
    x_vals = linspace(m_tlik(i) - sd_tlik(i), m_tlik(i) + sd_tlik(i), 100);
    y_vals = a(i) * x_vals + c(i);
    
    plot(x_vals, y_vals, 'Color',colour, 'LineWidth', 2)
    
    
    % plot Gaussians in transformed likelihood
    x_vals = linspace(m_tlik(i) - 3 * sd_tlik(i), ...
        m_tlik(i) + 3 * sd_tlik(i), 100);
    y_vals = normpdf(x_vals, m_tlik(i), sd_tlik(i)) ...
        * .02 * (mx - mn) * theta(3);
  
    plot(x_vals, y_vals, 'Color', colour);
    
    % plot approximate Gaussians in likelihood
    
    y_vals = linspace(m_lik(i) - 3 * sd_lik(i), m_lik(i) +  3 *sd_lik(i), 100);
    x_vals = normpdf(y_vals, m_lik(i), sd_lik(i)) ...
        * .01 * (mx - mn) * theta(3);
    
    plot(x_vals, y_vals, 'Color', colour);
    
    % plot exact distributions in likelihood
    
    y_vals = linspace(0.051, max(f(tliks)), 10000);
    ty_vals = invf(y_vals);
    x_vals = normpdf(ty_vals, m_tlik(i), sd_tlik(i)) ...
        ./ abs(theta(1) + 2 * theta(2) * ty_vals)...
          * 0.01 * (mx - mn) * theta(3);
    
    plot(x_vals, y_vals, '--', 'Color', colour);
    
    
    % indicate positions of these Gaussians in GPs over likelihood and
    % transformed likelihood
    
    figure(1)
    subplot(2, 1, 1)
    
    plot([xst(i), xst(i)], [m_tlik(i) - 2 * sd_tlik(i), m_tlik(i) + 2 * sd_tlik(i)], ...
        'LineWidth', 2, 'Color', colour);

    figure(1)
    subplot(2, 1, 2)
    
    plot([xst(i), xst(i)], [m_lik(i) - 2 * sd_lik(i), m_lik(i) + 2 * sd_lik(i)], ...
        'LineWidth', 2, 'Color', colour);
    
end

figure(3)
xlim([0, theta(3)/2]);
ylim([0, mx]);

set(0, 'defaulttextinterpreter', 'none')
matlabfrag([plotdir,'lik_v_tlik'])

figure(1)
matlabfrag([plotdir,'gps_lik_tlik'])

close all

function [f, df] = objective(logth, f_h, df_h, mn, mx, alpha, beta)

% maximum likelihood (or least squares) objective for our three constraints:
% exp(logth(3)) * df_h(0,exp(logth)) == alpha .* (mx - mn)
% f_h(exp(logth(3)), exp(logth)) == mx
% exp(logth(3)) * df_h(exp(logth(3)), exp(logth)) == beta .* (mx - mn)

f = (exp(logth(3)) * df_h(0,exp(logth)) - alpha .* (mx - mn)).^2 + ...
    (f_h(exp(logth(3)), exp(logth)) - mx).^2 + ...
    (exp(logth(3)) * df_h(exp(logth(3)), exp(logth)) - beta .* (mx - mn)).^2;

df = nan(3,1);

% these derivatives should ideally be keyed off input dtheta_f_h etc. input
% functions, but currently they are not
df(1) = 2 * exp(logth(1) + logth(3)) * ...
    (3 * exp(logth(3)) * (exp(logth(1)) + exp(logth(2) + logth(3))) ...
        + (mn - mx) * (1 + alpha + beta));
    
df(2) = 2 * exp(logth(1) + 2 * logth(3)) * ...
    (3 * exp(logth(1) + logth(3)) + 5 * exp(logth(2) + 2 * logth(3))  ...
        + (mn - mx) * (1 + 2 * beta));
    
df(3) = 2 * exp(logth(3)) * ...
    (10 * exp(2 * logth(2) + 3 * logth(3)) ...
    + exp(logth(1)) ...
        * (3 * exp(logth(1) + logth(3)) + (mn - mx) * (1 + alpha + beta)) ...
    + exp(logth(2) + logth(3)) ...
        * (9 * exp(logth(1) + logth(3)) + 2 * (mn - mx)*(1 + 2 * beta))...
    );

