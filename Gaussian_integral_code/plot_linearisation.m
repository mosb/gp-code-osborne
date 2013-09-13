function draw_from_linearisation

plotdir = '~/Docs/bayes-quadrature-for-gaussian-integrals-paper/figures/';

% b controls how wide the error bars are for large likelihoods
b = 1;
% a large likelihood is defined as having a transformed likelihood of 
% d * theta(1)
d = 1;

% a sets how wide the error bars are for small likelihoods
a = .1;
% a small likelihood is defined as having a transformed likelihood of 
% c * theta(1)
c = -1;


colours = cbrewer('qual','Paired', 12);
colours = colours(2:2:end, :);

% NB: results seem fairly insensitive to selection of a (sigh). b
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

theta2 = @(phi1) ...
    (-b*(1+4*c^2-2*d-4*c*d)*(mn-mx)+a*(-1-4*d^2+c*(2+4*d))*(mn-mx)...
      +2*(-c+2*c^2+d*(-1+2*d))*mx) ...
      / ...
      (2*exp(phi1)*(1+4*c^2-2*d+4*d^2-2*c*(1+2*d)));
theta3 = @(phi1) ...
    (b*(1+2*c-4*d)*(mn-mx)-a*(-1+4*c-2*d)*(mn-mx)-2*(-1+c+d)*mx)...
    /...
    (2 * exp(2*phi1) * (1+4*c^2-2*d+4*d^2-2*c*(1+2*d)));

f_h = @(tlik, phi1) theta2(phi1) * tlik + theta3(phi1) * tlik.^2;
df_h = @(tlik, phi1) theta2(phi1) + 2 * theta3(phi1) * tlik;
%ddf_h = @(tlik, theta) 2 * theta(3);
invf_h = @(lik, phi1) 0.5 * 1/theta3(phi1) * (...
            -theta2(phi1) + sqrt(theta2(phi1)^2 + 4 * theta3(phi1) * lik ));

options.MaxFunEvals = 1000;
[phi1, min_obj] = fminunc(@(phi1) objective(phi1, f_h, df_h, mn, mx, ...
    a, b, c, d), ...
    zeros(1,1), options);
min_obj

f = @(tlik) f_h(tlik, phi1);
df = @(tlik) df_h(tlik, phi1);
%ddf = @(tlik) ddf_h(tlik, theta);
invf = @(lik) invf_h(lik, phi1);

figure(3)
clf
n_tliks = 1000;
tliks = linspace(-exp(phi1), exp(phi1), n_tliks);
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
lin_slope = df(m_tlik);
lin_const = f(m_tlik) - lin_slope .* m_tlik;

% unnecessary under the quadratic transform
% % approximate linearisation
% best_tlik = mu;
% a = df(best_tlik) + (m_tlik - best_tlik) .* ddf(best_tlik);
% c = f(m_tlik) - a .* m_tlik;

% gp over likelihood
% ====================================

m_lik = diag(lin_slope) * m_tlik + lin_const;
C_lik =  diag(lin_slope) * C_tlik * diag(lin_slope);
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
    y_vals = lin_slope(i) * x_vals + lin_const(i);
    
    plot(x_vals, y_vals, 'Color',colour, 'LineWidth', 2)
    
    
    % plot Gaussians in transformed likelihood
    x_vals = linspace(m_tlik(i) - 3 * sd_tlik(i), ...
        m_tlik(i) + 3 * sd_tlik(i), 100);
    y_vals = normpdf(x_vals, m_tlik(i), sd_tlik(i)) ...
        * .02 * (mx - mn) * exp(phi1);
  
    plot(x_vals, y_vals, 'Color', colour);
    
    % plot approximate Gaussians in likelihood
    
    y_vals = linspace(m_lik(i) - 3 * sd_lik(i), m_lik(i) +  3 *sd_lik(i), 100);
    x_vals = normpdf(y_vals, m_lik(i), sd_lik(i)) ...
        * .01 * (mx - mn) * exp(phi1);
    
    plot(x_vals, y_vals, 'Color', colour);
    
    % plot exact distributions in likelihood
    
    y_vals = linspace(0, max(f(tliks)), 10000);
    ty_vals = invf(y_vals);
    x_vals = normpdf(ty_vals, m_tlik(i), sd_tlik(i)) ...
        ./ abs(theta(1) + 2 * theta(2) * ty_vals)...
          * 0.01 * (mx - mn) * exp;
    
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

% set(0, 'defaulttextinterpreter', 'none')
% matlabfrag([plotdir,'lik_v_tlik'])
% 
% figure(1)
% matlabfrag([plotdir,'gps_lik_tlik'])
% 
% close all

end

function [f] = objective(phi1, f_h, df_h, mn, mx, a, b, c, d)

% maximum likelihood (or least squares) objective for our three constraints:
% exp(logth(3)) * df_h(0,exp(logth)) == a .* (mx - mn)
% f_h(exp(logth(3)), exp(logth)) == mx
% exp(logth(3)) * df_h(exp(logth(3)), exp(logth)) == b .* (mx - mn)


f = (f_h(exp(phi1), phi1) - mx).^2 + ...
    (exp(phi1) * df_h(c * exp(phi1), phi1) - a .* (mx - mn)).^2 + ...
    (exp(phi1) * df_h(d * exp(phi1), phi1) - b .* (mx - mn)).^2;

%df = nan(3,1);

% these derivatives should ideally be keyed off input dtheta_f_h etc. input
% functions, but currently they are not
% df(1) = 2 * exp(x(1)) * ...
%     (2 * (1 + 4*c^2 + 4*d^2) * exp(3*x(1) + 2*x(3))...
%     + exp(x(1) + x(3)) * ((3 + 6*c + 6*d) * exp(x(1) + x(2)) ...
%         + 4 * (a*c + b*d) * (mn - mx) - 2*mx) ...
%     + exp(x(2)) * (3*exp(x(1)+x(2)) + (a+b)*mn - (1+a+b)*mx)...
%     );
    
% df(2) = 2 * exp(x(1) + x(2)) * ...
%     (3 * exp(x(1) + x(2)) + (1 + 2*c + 2*d) * exp(2*x(1) + x(3))  ...
%         + (a + b) * mn - (1 + a + b) * mx);
%     
% df(3) = 2 * exp(2*x(1) + x(3)) * ...
%     (...
%         (1 + 2*c + 2*d) * exp(x(1) + x(2)) ...
%         + (1 + 4*c^2 + 4*d^2) * exp(2*x(1) + x(3)) ...
%         + 2 * (a*c + b*d) * (mn - mx) - mx ...
%     );

end

