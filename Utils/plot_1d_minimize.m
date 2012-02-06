function [min_loss, next_sample_point] = ...
    plot_1d_minimize(objective_fn, bounds, samples)
% Optimizes a 1D function it by exhaustive evaluation,
% and plots the function as well.

    % Evaluate exhaustively between the bounds.
    N = 1000;
    test_pts = linspace(bounds(1), bounds(2), N);
    losses = nan(1, N);
    for loss_i=1:length(test_pts)
        losses(loss_i) = objective_fn(test_pts(loss_i));
    end

    % Choose the best point.
    [min_loss,min_ind] = min(losses);
    next_sample_point = test_pts(min_ind);

    % Plot the function.
    figure(1234); clf;
    h_surface = plot(test_pts, losses, 'b'); hold on;
    
    % Also plot previously chosen points.
    h_points = plot(samples.locations, arrayfun(objective_fn, samples.locations), ...
        'kd', 'MarkerSize', 4); hold on;
    h_best = plot(next_sample_point, min_loss, 'rd', 'MarkerSize', 4); hold on;
    xlabel('Sample location');
    ylabel('Expected uncertainty after adding a new sample');
    legend( [h_surface, h_points, h_best], {'Expected uncertainty', ...
        'Previously Sampled Points', 'Best new sample'}, 'Location', 'Best');
    drawnow;
end
