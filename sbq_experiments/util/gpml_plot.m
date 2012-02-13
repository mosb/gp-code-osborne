function h = gpml_plot( hypers, X, y )
%
% A helper function to plot the posterior of a 1D GP, given the hyperparams in the
% form of GPML hypers.

[N,D] = size(X);

figure;
h = plot( X, y, '.'); hold on;
x_limits = xlim;

xrange = linspace( x_limits(1), x_limits(2), 300 )';

inference = @infExact;
likfunc = @likGauss;
meanfunc = {'meanZero'};
max_iters = 1000;
covfunc = @covSEiso;

[f, variance] = gp( hypers, inference, meanfunc, covfunc, likfunc, X, y, xrange );

edges = [f+2*sqrt(variance); flipdim(f-2*sqrt(variance),1)]; 
hc1 = fill([xrange; flipdim(xrange,1)], edges, [6 6 8]/8, 'EdgeColor', 'none');
h1 = plot( xrange, f, 'b-', 'Linewidth', 2); hold on;
h2 = plot( X(1:N), y(1:N), 'kd', 'Linewidth', 2); hold on;

lengthscale = exp(hypers.cov(1));

y_limits = ylim;
y_scale = y_limits(2) - y_limits(1);
yval1 = y_limits(1) + 0.05.* y_scale;
x_loc = min(X);
line([x_loc, x_loc + lengthscale],[yval1,yval1], 'Color', 'k', 'Linewidth', 2);
line([x_loc, x_loc],[yval1 + 0.01*y_scale,yval1 - 0.01*y_scale], 'Color', 'k', 'Linewidth', 2);
line([x_loc + lengthscale, x_loc + lengthscale],[yval1 + 0.01*y_scale,yval1 - 0.01*y_scale], 'Color', 'k', 'Linewidth', 2);

yval2 = y_limits(1) + 0.1.* y_scale;
text( x_loc, yval2, 'lengthscale' );

legend_handle = legend( [h1 hc1 h2 ], {'GP Posterior Mean', 'GP Posterior Uncertainty', 'Data'}, 'Location', 'SouthEast');
%set_thesis_fonts( gca, legend_handle );
%set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
%set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$f(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 16);
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 16);
end

