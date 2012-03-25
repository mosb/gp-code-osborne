% Bare-bones Demo of Bayesian Quadrature with Log-GPs
%
% Show problems with the transform in the BBQ paper.
%
% Takes a function that's better modeled in the log-domain and tries to
% integrate under it.
%
% David Duvenaud
% Jan 2012
% ===========================


% Define a likelihood function (one that is well-modeled by a gp on the log).
likelihood = @(x)(normpdf(x,4,.4));
loglikelihood = @(x)log(likelihood(x));

% Plot likelihood.
N = 200;
xrange = linspace( 1.075, 6.25, N )';

clf;
col_width = 8.25381;  % ICML double column width in cm.
lw = 0.5;

% Choose function sample points.
n_f_samples = 3;
function_sample_points = [ 3.1 3.6 4.7];

% Evaluate likelihood and log-likelood at sample points.
y = likelihood(function_sample_points)';
logy = log(likelihood(function_sample_points)' + 1);

%pause;
myeps = eps;

% Model likelihood with a GP.
% =================================

% Define quadrature hypers.
quad_length_scale = .3;
quad_kernel = @(x,y)exp( - 0.5 * ( ( x - y ) .^ 2 ) ./ quad_length_scale );
quad_noise = 1e-6;

% Perform GP inference to get posterior mean function.
K = bsxfun(quad_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_noise^2 .* diag(ones(n_f_samples,1)) ); % Compute inverse covariance
weights = C * y;  % Now compute kernel function weights.
posterior = @(x)(bsxfun(quad_kernel, function_sample_points, x) * weights); % Construct posterior function.



% Model log-likelihood with a GP
% =================================
quad_log_length_scale = quad_length_scale * 4;
quad_log_kernel = @(x,y)exp( - 0.5 * (( x - y ) .^ 2 ) ./ quad_log_length_scale );
quad_log_noise = 1e-6;

K = bsxfun(quad_log_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_log_noise^2 .* diag(ones(n_f_samples,1)) ); % Compute inverse covariance
weights = C * logy;  % Now compute kernel function weights.
log_posterior = @(x)(bsxfun(quad_log_kernel, function_sample_points, x) * weights); % Construct posterior function.
log_posterior_variance = @(x)(bsxfun(quad_log_kernel, x, x) - diag((bsxfun(quad_log_kernel, x, function_sample_points) * C) * bsxfun(quad_log_kernel, function_sample_points, x)'));



% Without looking at the function again, model the difference between
% likelihood-GP posterior and exp(log-likelihood-GP posterior).
% =====================================================================

quad_diff_length_scale = quad_log_length_scale / 4;
quad_diff_kernel = @(x,y)exp( - 0.5 * (( x - y ) .^ 2 ) ./ quad_diff_length_scale );
quad_diff_noise = 1e-6;

% Choose surrogate function sample points.
n_diff_samples = 8;
diff_sample_points = [linspace( 2.2, 6.2, n_diff_samples) function_sample_points] ;
n_diff_samples = n_diff_samples + n_f_samples;
diff_values = log_posterior(diff_sample_points') - log(max(myeps,posterior(diff_sample_points') + 1));

K = bsxfun(quad_diff_kernel, diff_sample_points', diff_sample_points ); % Fill in gram matrix
C = inv( K + quad_diff_noise^2 .* diag(ones(n_diff_samples,1)) ); % Compute inverse covariance
weights = C * diff_values;  % Now compute kernel function weights.
delta = @(x)(bsxfun(quad_diff_kernel, diff_sample_points, x) * weights); % Construct posterior function.


% Final approximation is GP posterior plus exp(LGP posterior).
% ==============================================================
%final = @(x)(posterior(x) + diff_posterior(x));
final = @(x)(posterior(x).*(1 + delta(x)));


% Overworld
% ==========================
subaxis( 2, 1, 1,'SpacingVertical',0.1, 'MarginLeft', .1,'MarginRight',0);

like_handle = plot( xrange, likelihood(xrange), 'k'); hold on; % pause;
sp_handle = plot( function_sample_points, y, 'k.', 'Markersize', 10); hold on;
% Plot likelihood-GP posterior.
gpf_handle = plot( xrange, posterior(xrange), 'r'); hold on;

%exp_gpl_handle = plot( xrange, exp(log_posterior(xrange)), 'b-.'); hold on;

final_handle = plot( xrange, final(xrange), 'g-'); hold on;

legend( [ sp_handle, like_handle, gpf_handle, final_handle], ...
        { '$\ell(x_s)$', '$\ell(x)$', '$m(\ell(x))$', 'final approx' }, ...
        'Fontsize', 8, 'Location', 'EastOutside', 'Interpreter','latex');
legend boxoff  

set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$\ell(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', 8);
%set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
xlim([xrange(1) xrange(end)])




% Underworld
% ======================== 
subaxis( 2, 1, 2,'SpacingVertical',0.1, 'MarginLeft', .1,'MarginRight',0);

% Plot exp(likelihood-GP posterior).

log_like_handle = plot( xrange, log(likelihood(xrange) + 1), 'k'); hold on; % pause;
log_sp_handle = plot( function_sample_points, log(y), 'k.', 'Markersize', 10); hold on;
log_gpf_handle = plot( xrange, log(max(myeps,posterior(xrange) + 1)), 'r'); hold on;
gp_tl_handle = plot( xrange, log_posterior(xrange), 'b-.'); hold on;
diff_points_handle = plot( diff_sample_points, diff_values', 'ko','Markersize', 5); hold on;
delta_handle = plot( xrange, delta(xrange), 'b-'); hold on;

legend( [log_sp_handle, log_like_handle, log_gpf_handle, gp_tl_handle, diff_points_handle, delta_handle], ...
        { '$\log(\ell(x_s))$', '$\log(\ell(x))$', '$\log(m(\ell(x)))$', '$m(\log(\ell(x)))$', '$\log(\ell(x_c))$', '$m(\Delta(x))$' } ...
        , 'Fontsize', 8, 'Location', 'EastOutside', 'Interpreter','latex');
legend boxoff  

line( [xrange(1), xrange(end)], [0 0], 'linestyle', '--', 'color', 'k', 'linewidth', lw );
    
set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$log(\ell(x))$' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', 8);
%set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
xlim([xrange(1) xrange(end)])
%ylim([-1 4.5])


%position = get(gca, 'Position');          % Width of the figure
%tightinset = get(gca, 'TightInset');      % Width of the surrounding text
%total_width = position(3) + tightinset(3) + tightinset(1);
%scale_factor = 1;%(pagewidth*fraction)/total_width;
%set(gca, 'Position', [position(1:2), position(3:4).*scale_factor]);

set_fig_units_cm( 20, 20 );
%matlabfrag('~/Dropbox/papers/sbq-paper/figures/delta');  
savepng('bad2')

