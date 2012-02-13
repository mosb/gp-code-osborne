% Update our posterior uncertainty about the lengthscale hyperparameters.
% =========================================================================
if strcmp(opt.set_ls_var_method, 'laplace')
    % Set the variances of the lengthscales using the laplace
    % method around the maximum likelihood value.
    laplace_mode = gp_hypers.cov(1:end - 1);

    % Specify the likelihood function which we'll be taking the hessian of:
    % Todo: check that there isn't a scaling factor since Mike uses
    %       Gaussians, and Carl uses sqexp.
    %       Also, there is the matter of scaling the values.
    % Todo: Compute the Hessian using the gradient instead.
    like_func = @(log_in_scale) gpml_likelihood( log_in_scale, gp_hypers, ...
        inference, meanfunc, covfunc, likfunc, samples.locations, ...
        samples.scaled_r);

    % Find the Hessian.
    laplace_sds = Inf;
    try
        laplace_sds = sqrt(-1./hessdiag( like_func, laplace_mode));
    catch e; e; end

    % A little sanity checking, since at first the length scale won't be
    % sensible.
    bad_sd_ixs = isnan(laplace_sds) | isinf(laplace_sds) | (abs(imag(laplace_sds)) > 0);
    if any(bad_sd_ixs)
        warning(['Infinite or positive lengthscales, ' ...
                'Setting lengthscale variance to prior variance']);
        good_sds = sqrt(diag(prior.covariance));
        laplace_sds(bad_sd_ixs) = good_sds(bad_sd_ixs);
    end
    opt.sds_tr_input_scales = laplace_sds;

    if opt.plots && D == 1
        plot_hessian_approx( like_func, laplace_sds, laplace_mode );
    end
end
