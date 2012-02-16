% A simple cartoon of Bayesian Monte Carlo.
%
% David Duvenaud
% February 2012
% ===========================


function bmc_intro



% Plot our function.
N = 200;
xrange = linspace( 0, 25, N )';

% Choose function sample points.
function_sample_points = [ 5 12 16 ];
y = [ 2 8 4]';


% Model function with a GP.
% =================================

% Define quadrature hypers.
quad_length_scale = 2;
quad_kernel = @(x,y)exp( -0.5 * ( ( x - y ) .^ 2 ) ./ exp(quad_length_scale) );
quad_kernel_dl = @(x,y)( -0.5 * ( ( x - y ) .^ 2 ) .* quad_kernel(x, y) ) ./ exp(quad_length_scale);
quad_kernel_at_data = @(x)(bsxfun(quad_kernel, x, function_sample_points));
quad_kernel_dl_at_data = @(x)(bsxfun(quad_kernel_dl, x, function_sample_points));
quad_noise = 1e-6;

% Perform GP inference to get posterior mean function.
K = bsxfun(quad_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_noise^2 .* diag(N) ); % Compute inverse covariance
weights = C * y;  % Now compute kernel function weights.
posterior = @(x)(bsxfun(quad_kernel, function_sample_points, x) * weights); % Construct posterior function.
posterior_variance = @(x)(bsxfun(quad_kernel, x, x) - diag((bsxfun(quad_kernel, x, function_sample_points) * C) * bsxfun(quad_kernel, function_sample_points, x)'));
K_dl = bsxfun(quad_kernel_dl, function_sample_points, function_sample_points');
dmu_dl = @(x)( ( quad_kernel_dl_at_data(x) * C - quad_kernel_at_data(x) * C * K_dl * C )) * y;  % delta in the mean function vs delta in lengthscale.


% Plot posterior variance.
clf;
edges = [posterior(xrange)+2*sqrt(posterior_variance(xrange)); flipdim(posterior(xrange)-2*sqrt(posterior_variance(xrange)),1)];
hc1 = fill([xrange; flipdim(xrange,1)], edges, [6 6 8]/8, 'EdgeColor', 'none'); hold on;

[h,g] = crosshatch_poly([xrange; flipdim(xrange,1)], [posterior(xrange); zeros(size(xrange))], -45, 1, ...
    'linestyle', '-', 'linecolor', 'k', 'linewidth', 1, 'hold', 1);
fill( [xrange; flipdim(xrange,1)], [posterior(xrange); 10.*ones(size(xrange))], [ 1 1 1], 'EdgeColor', 'none');

edges = [posterior(xrange)+2*sqrt(posterior_variance(xrange)); flipdim(posterior(xrange),1)];
hc1 = fill([xrange; flipdim(xrange,1)], edges, [6 6 8]/8, 'EdgeColor', 'none'); hold on;


h1 = plot( xrange, posterior(xrange), 'b-', 'Linewidth', 1); hold on;
h2 = plot( function_sample_points, y, 'kd', 'Marker', 'd', ...
 'MarkerSize', 1.5, 'Linewidth', 1 );
 %'Color', [0.6 0.6 0.6]..

% Add axes, legend, make the plot look nice, and save it.
xlim( [xrange(1) - 0.04, xrange(end)]);
ylim( [ -8 10] );
legend( [h2 h1 hc1 g(1)], {'samples', 'posterior mean', 'posterior variance', 'expected area'}, 'Location', 'SouthEast', 'Fontsize', 6);
set( gca, 'XTick', [] );
set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$\ell(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
legend boxoff

set_fig_units_cm( 8, 4 );
matlabfrag('~/Dropbox/papers/sbq-paper/figures/bmc_intro');
%savepng('int_hypers');
%saveeps('int_hypers');

if 0

%figure
%plot(posterior_variance(xrange)); hold on;
%plot(c_theta0, 'g')

% Actually integrate lengthscale.
theta_mu = quad_length_scale;
theta_sigma = sqrt(varscale);

num_outer_samples = 100;
num_inner_samples = 10;
cdf_vals = linspace(0,1, num_outer_samples + 2);
cdf_vals = cdf_vals(2:end-1);  % don't include 0 and 1

cur_sample = 1;
f_history = NaN(num_outer_samples * num_inner_samples, length(xrange));
f_mu_history = NaN(num_outer_samples * num_inner_samples, length(xrange));

for i = 1:length(cdf_vals)

    % sample a lengthscale
    cur_lengthscale = norminv( cdf_vals(i), theta_mu, theta_sigma );
    
    % Recalculate process posterior.
    quad_kernel = @(x,y)exp( -0.5 * ( ( x - y ) .^ 2 ) ./ exp(cur_lengthscale) );
    K = bsxfun(quad_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
    Cinv = K + quad_noise^2 .* diag(N);
    weights = Cinv \ y;  % Now compute kernel function weights.

    posterior = @(x)(bsxfun(quad_kernel, function_sample_points, x) * weights); % Construct posterior function.
    posterior_covariance = @(x)(bsxfun(quad_kernel, x', x) - (bsxfun(quad_kernel, x, function_sample_points) / Cinv) * bsxfun(quad_kernel, function_sample_points, x)');
    
    % Evaluate posterior mean and covariance.
    fmu = posterior(xrange);
    fsigma = posterior_covariance(xrange);
      
    fprintf('.');  % Progress.
    
    for j = 1:num_inner_samples
        f_history(cur_sample, :) = mvnrnd(fmu, fsigma);
        f_mu_history(cur_sample, : ) = fmu;
        cur_sample = cur_sample + 1;
    end
end

figure;
plot(f_mu_history')
set(gcf, 'Units', 'centimeters');
set(gcf, 'Position', [1, 1, 30, 15]);
%set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
%set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$f(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 16);
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 16);
title('Posterior means when integrating over lengthscales');
savepng('means');

figure;
mike = plot(2.*sqrt(extra_var), 'b', 'Linewidth', 2 ); hold on;
emp = plot(2.*std(f_mu_history), 'g', 'Linewidth', 2 ); hold on;
combined_est = plot(2.*sqrt(posterior_variance(xrange) + extra_var), 'k-', 'Linewidth', 2 ); hold on;
combined_truth = plot(2.*std(f_history)', 'r-', 'Linewidth', 2 ); hold on;
set(gcf, 'Units', 'centimeters');
set(gcf, 'Position', [1, 1, 30, 15]);
%set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
%set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$f(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 16);
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 16);
title('Different variance contributions');
legend( [emp mike combined_truth combined_est], {'Empirical variance due to change in mean', 'Linear approx to variance due to change in mean', 'Empirical total posterior variance', 'Approximate total posterior variance'}, 'Location', 'SouthEast');
savepng('variances');
end
end


