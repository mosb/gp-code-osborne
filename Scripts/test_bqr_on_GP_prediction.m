clear
num_data = 100;
X_data = rand(num_data,5);

y_fn = @(x) 10*sin(pi*x(:,1).*x(:,2)) ...
            + 20*(x(:,3)-0.5).^2 ...
            + 10*x(:,4) ...
            + 5*x(:,5);
y_data = y_fn(X_data) + randn(num_data,1);

gp = set_gp('sqdexp','constant', [], X_data, y_data, ...
    1);

for i = 1:numel(gp.hyperparams)
    gp.hyperparams(i).priorMean = 0;
    gp.hyperparams(i).priorSD = 2;
    % NB: R&Z put prior on noise and signal VARIANCE; we place prior on
    % standard deviation.
end

prior_means = vertcat(gp.hyperparams(:).priorMean);
prior_sds = vertcat(gp.hyperparams(:).priorSD);

p_fn = @(x) mvnpdf(x, prior_means, diag(prior_sds.^2));
r_fn = @(x) exp(log_gp_lik(sample, X_data, y_data, gp));

p_r_fn = @(x) p_fn(x) * r_fn(x);
        
max_num_samples = 500;
samples = slicesample(prior.means, max_num_samples,...
    'pdf', p_r_fn,'width', prior_sds);

for i = 1:num_samples
    
    
    gp.hypersamples(i).hyperparameters = new_sample;

    gp = revise_gp(X_data, y_data, gp, 'overwrite',[], i);

    
        opt.print = false;
        opt.optim_time = 60;

        gp = train_gp('sqdexp', 'constant', [], samples, r, opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gp);
        
        r_gp.quad_output_scale = best_hypersample_struct.output_scale;
        r_gp.quad_input_scales = best_hypersample_struct.input_scales;
        r_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
   
        [BQR(dims_ind, trial), BQ(dims_ind, trial)] = ...
            predict(sample_struct, prior, r_gp);
                
 
end