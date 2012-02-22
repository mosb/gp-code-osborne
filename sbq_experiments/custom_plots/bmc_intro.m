% A simple cartoon of Bayesian Monte Carlo.
%
% David Duvenaud
% February 2012
% ===========================


function bmc_intro


col_width = 8.25381;  % ICML double column width in cm.
lw = .5;  % line width
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

[h,g] = crosshatch_poly([xrange; flipdim(xrange,1)], [posterior(xrange); zeros(size(xrange))], -30, 1, ...
    'linestyle', '-', 'linecolor', [ 0 .8 0 ], 'linewidth', lw, 'hold', 1);
fill( [xrange; flipdim(xrange,1)], [posterior(xrange); 10.*ones(size(xrange))], [ 1 1 1], 'EdgeColor', 'none');

edges = [posterior(xrange)+2*sqrt(posterior_variance(xrange)); flipdim(posterior(xrange),1)];
hc1 = fill([xrange; flipdim(xrange,1)], edges, [6 6 8]/8, 'EdgeColor', 'none'); hold on;


h1 = plot( xrange, posterior(xrange), 'b-', 'Linewidth', lw); hold on;
h2 = plot( function_sample_points, y, 'kd', 'Marker', '.', ...
 'MarkerSize', 5, 'Linewidth', lw );
 %'Color', [0.6 0.6 0.6]..

% Add axes, legend, make the plot look nice, and save it.
xlim( [xrange(1) - 0.04, xrange(end)]);
ylim( [ -8 10] );
legend( [h2 h1 hc1 g(1)], ...
        {'samples', 'posterior mean', 'posterior variance', 'expected area'}, ...
        'Location', 'SouthEast', 'Fontsize', 6);
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

set_fig_units_cm( col_width, 4 );
matlabfrag('~/Dropbox/papers/sbq-paper/figures/bmc_intro');
%savepng('int_hypers');
%saveeps('int_hypers');

end


