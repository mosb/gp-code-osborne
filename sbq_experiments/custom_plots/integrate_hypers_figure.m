% Bare-bones Demo of Bayesian Quadrature when integrating out hypers,
% assuming that the posterior mean is linear in the hypers, and that
% the posterior distribution over hypers is normal, centered at the point used
% to calculate the postrior mean.
%
% Written to help understand Mike Osborne's AISTATS paper.
%
% David Duvenaud
% Jan 2012
% ===========================


function approx_lengthscale_integrate_demo



% Plot our function.
N = 200;
xrange = linspace( -2, 30, N )';

% Choose function sample points.
function_sample_points = [ 0 6 12 16 ];
y = [ 1 2 0 2]';


% Model function with a GP.
% =================================

% Define quadrature hypers.
quad_length_scale = 1.5;
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

% Check derivative
%quad_kernel2 = @(x,y)exp( -0.5 * ( ( x - y ) .^ 2 ) ./ exp(quad_length_scale + 0.00001) );
%K2 = bsxfun(quad_kernel2, function_sample_points', function_sample_points ); % Fill in gram matrix
%C2 = inv( K2 + quad_noise^2 .* diag(N) ); % Compute inverse covariance
%weights2 = C2 * y;  % Now compute kernel function weights.
%posterior2 = @(x)(bsxfun(quad_kernel2, function_sample_points, x) * weights2); % Construct posterior function.
%t = (posterior(xrange) - posterior2(xrange)) ./ 0.00001;

% Evaluate gradient of mean w.r.t. lengthscale at each point.
c_theta0 = dmu_dl(xrange);
varscale = 1;  % proportional to posterior variance in lengthscale;
extra_var = c_theta0.^2 .* varscale;


% Plot posterior variance.
figure;
edges = [posterior(xrange)+2*sqrt(posterior_variance(xrange)); flipdim(posterior(xrange)-2*sqrt(posterior_variance(xrange)),1)];
edges2 = [posterior(xrange)+2*sqrt(posterior_variance(xrange) + extra_var); flipdim(posterior(xrange)-2*sqrt(posterior_variance(xrange) + extra_var),1)];
hc2 = fill([xrange; flipdim(xrange,1)], edges2, [6 8 6]/8, 'EdgeColor', 'none'); hold on;
hc1 = fill([xrange; flipdim(xrange,1)], edges, [6 6 8]/8, 'EdgeColor', 'none'); hold on;
h1 = plot( xrange, posterior(xrange), 'b-', 'Linewidth', 2); hold on;
h2 = plot( function_sample_points, y, '.', 'Marker', '.', ...
 'MarkerSize', 10, ...
 'Color', [0.6 0.6 0.6]); hold on;

% Add axes, legend, make the plot look nice, and save it.

set(gcf, 'Units', 'centimeters');
set(gcf, 'Position', [1, 1, 30, 15]);
xlim( [xrange(1) - 0.04, xrange(end)]);
legend_handle = legend( [h2 h1 hc1 hc2], {'Data', 'GP Posterior Mean', 'GP Posterior Uncertainty', 'GP Posterior Uncertainty integrating lengthscale '}, 'Location', 'SouthEast');
%set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
%set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$f(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 16);
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 16);


set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
%set(fh, 'color', 'white');
set(gca, 'YGrid', 'off');
legend boxoff


savepng('test');
drawnow;
pause(0.001);


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

function savepng( name )
    handle = gcf;
    % Make changing paper type possible
    set(handle,'PaperType','<custom>');

    % Set units to all be the same
    set(handle,'PaperUnits','inches');
    set(handle,'Units','inches');

    % Set the page size and position to match the figure's dimensions
    paperPosition = get(handle,'PaperPosition');
    position = get(handle,'Position');
    set(handle,'PaperPosition',[0,0,position(3:4)]);
    set(handle,'PaperSize',position(3:4));
    print( gcf, sprintf('-r%d', ceil(72*(4/3))), '-dpng', [ name '.png'] );
end
