function [min_loss, next_sample_point] = ...
    plot_1d_minimize(objective_fn, bounds, samples, log_var_ev)
% Optimizes a 1D function it by exhaustive evaluation,
% and plots the function as well.

    % Evaluate exhaustively between the bounds.
    N = 1000;
    test_pts = linspace(bounds(1), bounds(2), N);
    losses = nan(1, N);
    m = nan(1, N);
    V = nan(1, N);
    for loss_i=1:length(test_pts)
        [losses(loss_i), m(loss_i), V(loss_i)] = objective_fn(test_pts(loss_i));
    end

    % Choose the best point.
    [min_loss,min_ind] = min(losses);
    next_sample_point = test_pts(min_ind);
    
    % make a plot of the gp
    figure(666);clf
    gamma = (exp(1) - 1)^(-1);
    tr = log(exp(samples.log_r-max(samples.log_r))/gamma + 1);
    gp_plot(test_pts, m, sqrt(V), samples.locations, tr);


    
    % Plot the function.
    figure(1234); clf;
    h_surface = plot(test_pts, losses, 'b'); hold on;
    
    % Plot existing neg-sqd-mean-ev
    nsme = exp(log_var_ev);
    h_exist = plot(bounds, [nsme nsme], 'k');
    
    % Also plot previously chosen points.
    h_points = plot(samples.locations, arrayfun(objective_fn, samples.locations), ...
        'kd', 'MarkerSize', 4); hold on;
    h_best = plot(next_sample_point, min_loss, 'rd', 'MarkerSize', 4); hold on;
    xlabel('Sample location');
    ylabel('Expected variance after adding a new sample');
    legend( [h_surface, h_points, h_best, h_exist], {'Expected uncertainty', ...
        'Previously Sampled Points', 'Best new sample', 'existing variance'}, 'Location', 'Best');
    drawnow;
end
