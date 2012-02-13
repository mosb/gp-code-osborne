function plot_hessian_approx( like_func, laplace_sds, x0 )

    % Plot the log-likelihood surface.
    figure(11); clf;
    hrange = linspace(-10, 10, 1000 );
    for t = 1:length(hrange)
        vals(t) = like_func(hrange(t));
    end
    actual=plot(hrange, vals, 'b'); hold on;
    y=get(gca,'ylim');
    h=plot([x0 x0],y, 'g');

    % Plot the laplace-approx Gaussian.
    rescale = like_func(x0)/mvnpdf(0, 0, laplace_sds^2);
    laplace_h=plot(hrange, rescale.*mvnpdf(hrange', x0, laplace_sds^2), 'r');
    xlabel('log input scale');
    ylabel('likelihood');
end
