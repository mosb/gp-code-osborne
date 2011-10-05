cd ~/Code/gp-code-osborne/
addpath(genpath('~/Code/gp-code-osborne/'))
addpath ~/Code/lightspeed
addpath(genpath('~/Code/Utils/'))
rmpath ~/Code/CoreGP
rmpath ~/Code/BQR


clear
num_data = 100;
num_star = 100;

opt.print = false;
opt.optim_time = 30;
opt.num_hypersamples = 25;
opt.noiseless = true;

max_num_samples = 500;


X_data = rand(num_data,5);

y_fn = @(x) 10*sin(pi*x(:,1).*x(:,2)) ...
            + 20*(x(:,3)-0.5).^2 ...
            + 10*x(:,4) ...
            + 5*x(:,5);
y_data = y_fn(X_data) + randn(num_data,1);

X_star = rand(num_star, 5);
y_star = y_fn(X_star);

save prob_bqr_on_GP_prediction;

gp = set_gp('sqdexp','constant', [], X_data, y_data, ...
    1);

% we do not marginalise over priorMean
active_hp_inds = 1:7;
gp.active_hp_inds = active_hp_inds;


for i = 1:numel(gp.hyperparams)
    gp.hyperparams(i).priorMean = 0;
    gp.hyperparams(i).priorSD = 2;
    % NB: R&Z put prior on noise and signal VARIANCE; we place prior on
    % standard deviation.
end

prior_means = vertcat(gp.hyperparams(active_hp_inds).priorMean);
prior_sds = vertcat(gp.hyperparams(active_hp_inds).priorSD);
prior.means = prior_means;
prior.sds = prior_sds;

p_fn = @(x) mvnpdf(x, prior_means', diag(prior_sds.^2));
r_fn = @(x) exp(log_gp_lik(x, X_data, y_data, gp));

p_r_fn = @(x) p_fn(x) * r_fn(x);
        
%for trial = 1:num_trials
    samples = slicesample(prior_means', max_num_samples,...
        'pdf', p_r_fn,'width', prior_sds');

    for i = 1:max_num_samples
        gp.hypersamples(i).hyperparameters(active_hp_inds) = samples(i,:);
        gp.hypersamples(i).hyperparameters(8) = mean(y_data);
    end
    gp = revise_gp(X_data, y_data, gp, 'overwrite');
    r = vertcat(gp.hypersamples.logL);

    mean_y = nan(num_star, max_num_samples);
    var_y = nan(num_star, max_num_samples);
    for hs = 1:max_num_samples
        [mean_y(:, hs), var_y(:, hs)] = ...
            posterior_gp(X_star,gp,hs,'var_not_cov');
    end

    mean_y = mean_y';
    var_y = var_y';

    qd = mean_y;
    qdd = var_y + mean_y.^2;

    gpr = [];
    gpqd = [];
    gpqdd = [];

    MC_mean = nan(num_star,max_num_samples);
    BMC_mean = nan(num_star,max_num_samples);
    BQ_mean = nan(num_star,max_num_samples);
    BQR_mean = nan(num_star,max_num_samples);

    MC_sd = nan(num_star,max_num_samples);
    BMC_sd = nan(num_star,max_num_samples);
    BQ_sd = nan(num_star,max_num_samples);
    BQR_sd = nan(num_star,max_num_samples);

    perf_MC = nan(1,max_num_samples);
    perf_BMC = nan(1,max_num_samples);
    perf_BQ = nan(1,max_num_samples);
    perf_BQR = nan(1,max_num_samples);


    warning('off','revise_gp:small_num_data');

    for i = 1:max_num_samples

        samples_i = samples(1:i, :);
        r_i = r(1:i, :);
        qd_i = qd(1:i, :);
        qdd_i = qdd(1:i, :);




        sample_struct = struct();
        sample_struct.samples = samples_i;
        sample_struct.log_r = log(r_i);
        sample_struct.qd = qd_i;
        sample_struct.qdd = qdd_i;     

        opt.optim_time = 30;
        opt.active_hp_inds = 2:9;
        opt.prior_mean = 0;
        
        gpr = train_gp('sqdexp', 'constant', gpr, ...
            samples_i, r_i, opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpr);

        r_gp.quad_output_scale = best_hypersample_struct.output_scale;
        r_gp.quad_input_scales = best_hypersample_struct.input_scales;
        r_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        r_gp.quad_mean = best_hypersample_struct.mean;

        opt.prior_mean = 'default';
        gpqd = train_gp('sqdexp', 'constant', gpqd, ...
            samples_i, qd_i(:,1), opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpqd);

        qd_gp.quad_output_scale = best_hypersample_struct.output_scale;
        qd_gp.quad_input_scales = best_hypersample_struct.input_scales;
        qd_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        qd_gp.quad_mean = mean(qd_i,1);

        opt.prior_mean = 'default';
        gpqdd = train_gp('sqdexp', 'constant', gpqdd, ...
            samples_i, qdd_i(:,1), opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpqdd);

        qdd_gp.quad_output_scale = best_hypersample_struct.output_scale;
        qdd_gp.quad_input_scales = best_hypersample_struct.input_scales;
        qdd_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        
        opt.optim_time = 3;
        opt.active_hp_inds = [];
        opt.prior_mean = 'train';
        for num_star = 1:num_star
            gpqdd_star = train_gp('sqdexp', 'constant', gpqdd, ...
                samples_i, qdd_i(:,num_star), opt);
            [best_hypersample, best_hypersample_struct] = ...
                disp_hyperparams(gpqdd_star);

            qdd_gp.quad_mean(num_star) = best_hypersample_struct.mean;
        end

        [BQR_mean(:,i), BQR_sd(:,i), BQ_mean(:,i), BQ_sd(:,i)] = ...
            predict(sample_struct, prior, r_gp, qd_gp, qdd_gp);

        [BMC_mean(:,i), BMC_sd(:,i)] = predict_BMC(sample_struct, prior, r_gp);

        [MC_mean(:,i), MC_sd(:,i)] = predict_MC(sample_struct, prior);

        perf_BQR(i) = performance(X_star,BQR_mean(:,i),BQR_sd(:,i),X_star,y_star);
        perf_BQ(i) = performance(X_star,BQ_mean(:,i),BQ_sd(:,i),X_star,y_star);
        perf_BMC(i) = performance(X_star,BMC_mean(:,i),BMC_sd(:,i),X_star,y_star);
        perf_MC(i) = performance(X_star,MC_mean(:,i),MC_sd(:,i),X_star,y_star);

        fprintf('Sample %u\n performance\n BQR:\t%g\n BQ:\t%g\n BMC:\t%g\n MC:\t%g\n',...
            i,perf_BQR(i),perf_BQ(i),perf_BMC(i),perf_MC(i));


        save test_bqr_on_GP_prediction
    end
%end


% figure;hold on;
% plot(perf_BQR,'r')
% plot(perf_BMC,'b')
% plot(perf_MC,'m')