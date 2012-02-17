function gen_eue_plot
%
% Makes a plot showing a sequence of expected uncertainty surfaces.
%
% David Duvenaud
% February 2012
% ============================

col_width = 8.25381;  % ICML double column width in cm.

clf;
randn('state', 0);
rand('twister', 0);  

prior.mean = 0;
prior.covariance = 1;

log_likelihood_fn = @(x)logmvnpdf( x, -.35, 3 ); 

D = 1;

% Set unspecified fields to default values.
opt = struct('num_samples', 20, ...
             'gamma', .1, ...
             'set_ls_var_method', 'laplace');
opt.num_samples = 18;
opt.plots = false;

% Initialize with some random points.
for i = 1:2
next_sample_point = i - 2;
samples.locations(i,:) = next_sample_point;
samples.log_l(i,:) = log_likelihood_fn(next_sample_point);
end
next_sample_point = 1.9;

for i = size(samples.log_l,1) + 1:opt.num_samples

    % Update sample struct.
    % ==================================
    samples.locations(i,:) = next_sample_point;          % Record the current sample location.
    samples.log_l(i,:) = log_likelihood_fn(next_sample_point);   % Sample the integrand at the new point.
    samples.max_log_l = max(samples.log_l); % all log-likelihoods have max_log_l subtracted off
    samples.scaled_l = exp(samples.log_l - samples.max_log_l);
    samples.tl = log_transform(samples.scaled_l, opt.gamma);

    % Train GPs
    % ===========================   
    inference = @infExact;
    likfunc = @likGauss;
    meanfunc = {'meanZero'};
    max_iters = 1000;
    covfunc = @covSEiso;

    % Init GP Hypers.
    init_hypers.mean = [];
    init_hypers.lik = log(0.01);  % Values go between 0 and 1, so no need to scale.
    init_lengthscales = mean(sqrt(diag(prior.covariance)))/10;
    init_output_variance = .1;
    init_hypers.cov = log( [init_lengthscales init_output_variance] ); 

    % Fit the model, but not the likelihood hyperparam (which stays fixed).
    fprintf('Fitting GP to observations...\n');
    gp_hypers = init_hypers;
    gp_hypers = minimize(gp_hypers, @gp_fixedlik, -max_iters, ...
                         inference, meanfunc, covfunc, likfunc, ...
                         samples.locations, samples.scaled_l);
    if any(isnan(gp_hypers.cov))
        gp_hypers = init_hypers;
        warning('Optimizing hypers failed');
    end
    l_gp_hypers.log_output_scale = gp_hypers.cov(end);
    l_gp_hypers.log_input_scales(1:D) = gp_hypers.cov(1:end - 1);
    fprintf('Output variance: '); disp(exp(l_gp_hypers.log_output_scale));
    fprintf('Lengthscales: '); disp(exp(l_gp_hypers.log_input_scales));

    fprintf('Fitting GP to log-observations...\n');
    gp_hypers_log = init_hypers;
    gp_hypers_log = minimize(gp_hypers_log, @gp_fixedlik, -max_iters, ...
                             inference, meanfunc, covfunc, likfunc, ...
                             samples.locations, samples.tl);        
    if any(isnan(gp_hypers_log.cov))
        gp_hypers_log = init_hypers;
        warning('Optimizing hypers on log failed');
    end
    tl_gp_hypers.log_output_scale = gp_hypers_log.cov(end);
    tl_gp_hypers.log_input_scales(1:D) = gp_hypers_log.cov(1:end - 1);
    fprintf('Output variance of logL: '); disp(exp(tl_gp_hypers.log_output_scale));
    fprintf('Lengthscales on logL: '); disp(exp(tl_gp_hypers.log_input_scales));

    if opt.plots
        figure(50); clf;
        gpml_plot( gp_hypers, samples.locations, samples.scaled_l);
        title('GP on untransformed values');
        figure(51); clf;
        gpml_plot( gp_hypers_log, samples.locations, samples.tl);
        title('GP on log( exp(scaled) + 1) values');
    end

    % Optionally set uncertainty in lengthscales using the laplace
    % method around the maximum likelihood value.
    if strcmp(opt.set_ls_var_method, 'laplace')
        % Specify the likelihood function which we'll be taking the hessian of:
        like_func = @(log_in_scale) gpml_lengthscale_likelihood( log_in_scale, ...
            gp_hypers_log, inference, meanfunc, covfunc, likfunc, ...
            samples.locations, samples.tl);

        laplace_mode = gp_hypers_log.cov(1:end - 1);
        failsafe_sds = sqrt(diag(prior.covariance));
        opt.sds_tl_log_input_scales = ...
            likelihood_laplace( like_func, laplace_mode, failsafe_sds);

        if opt.plots && D == 1
            plot_hessian_approx( like_func, opt.sds_tl_log_input_scales, laplace_mode );
        end
    else
        opt.sds_tl_log_input_scales = false;
    end

    [log_mean_evidence, log_var_evidence, ev_params, del_gp_hypers] = ...
        log_evidence(samples, prior, l_gp_hypers, tl_gp_hypers, [], opt);

    % Choose the next sample point.
    % =================================

    % Define the criterion to be optimized.
    objective_fn = @(new_sample_location) expected_uncertainty_evidence...
            (new_sample_location(:)', samples, prior, ...
            l_gp_hypers, tl_gp_hypers, del_gp_hypers, ev_params, opt);

    % Define the box with which to bound the selection of samples.
    lower_bound = prior.mean - 5*sqrt(diag(prior.covariance))';
    upper_bound = prior.mean + 5*sqrt(diag(prior.covariance))';
    bounds = [lower_bound; upper_bound]';            

  % Evaluate exhaustively between the bounds.
    N = 100;
    test_pts(:,i) = linspace(bounds(1), bounds(2), N);

    for loss_i=1:length(test_pts)
        [losses(loss_i,i)] = objective_fn(test_pts(loss_i, i));
    end

    % Choose the best point.
    [exp_loss_min(i),min_ind(i)] = min(losses(:,i));
    next_sample_point = test_pts(min_ind(i), i);
    chosen_point(i) = next_sample_point;

     for loss_i=1:i
        point_losses(loss_i,i) = objective_fn(samples.locations(i));
    end

    nsme(i) = exp(log_var_evidence);   % existing neg-sqd-mean-ev
end



% Plotting
% ===========================

cmap = colormap;
clf;
for i = 4:opt.num_samples
    end_ix = opt.num_samples;
    cur_color = cmap(floor(size(cmap, 1)*( 1- i/end_ix)) + 1, :);
    h_est = plot3(repmat(i,N,1), test_pts(:,i), losses(:,i), ...
        'Color', cur_color, 'LineWidth', 1); hold on;
    
    % Also plot previously chosen points.
    h_points = plot3(repmat(i,i,1), samples.locations(1:i), point_losses(1:i, i), ...
        'k.', 'MarkerSize', 5); hold on;
    h_best = plot3(i, chosen_point(i), exp_loss_min(i), 'r.', 'MarkerSize', 10); hold on;
end

xrange = test_pts(:,end);

% Plot the prior.
scale_factor = max(losses(:)) / mvnpdf(0, 0, prior.covariance);
h_prior = plot3(repmat(i + 1,N,1), xrange,...
    mvnpdf(xrange, prior.mean, prior.covariance) * scale_factor, 'k', 'LineWidth', 1); hold on;

% Plot the true function.
true_plot_depth = i + 1;
scale_factor = max(losses(:)) / max(exp(log_likelihood_fn(xrange)));
h_ll = plot3(repmat(true_plot_depth,N,1), xrange, ...
    scale_factor * exp(log_likelihood_fn(xrange)), 'g', 'LineWidth', 2);

%legend( [ h_prior h_ll h_est SDh h_samples ], {'Prior', 'True Likelihood Function', 'Posterior mean', 'Marginal uncetainty', 'Sample locations'}, 'Location', 'Best');
legend boxoff     
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(gcf, 'color', 'white'); 
set(gca, 'YGrid', 'off');  

grid on;
%zlim( [ 0 .5  ] );
%xlim( [ 0 true_plot_depth ] );
%title('Active learning of an integral');
xlabel('samples');
ylabel('x');
zlabel('f(x)');
view(-82, 48);

set_fig_units_cm( col_width, 7 );
matlabfrag('~/Dropbox/papers/sbq-paper/figures/eue_progression');  
end
