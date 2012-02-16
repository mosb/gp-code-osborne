function plot_hessian_approx( like_func, laplace_sds, x0 )

    % Plot the log-likelihood surface.
    figure(11); clf;
    hrange = linspace(-10, 10, 1000 );
    for t = 1:length(hrange)
        vals(t) = like_func(hrange(t));
    end
    actual=plot(hrange, vals, 'k'); hold on;
    y=get(gca,'ylim');
    h_peak=plot([x0 x0],y, 'b--');

    % Plot the laplace-approx Gaussian.
    rescale = like_func(x0)/mvnpdf(0, 0, laplace_sds^2);
    laplace_h=plot(hrange, rescale.*mvnpdf(hrange', x0, laplace_sds^2), 'b');
    xlabel('log input scale');
    ylabel('likelihood');
    title('Lengthscale laplace approximation');

        set(gca, 'TickDir', 'out')
    set(gca, 'Box', 'off', 'FontSize', 10); 
    set(gcf, 'color', 'white'); 
    set(gca, 'YGrid', 'off');
    legend([ actual, laplace_h], {'actual likelihood surface', 'laplace approximation'});
    legend boxoff

end
