function [gp, quad_gp] = train_gpgo(gp, X_data, y_data, opt)
% train gp for gpgo

warning off

% switch derivatives on now.
if isfield(gp, 'grad_hyperparams')
    grad_hyperparams = gp.grad_hyperparams;
else
    grad_hyperparams = true;
end
gp.grad_hyperparams = true;

r_X_data = vertcat(gp.hypersamples.hyperparameters);
r_y_data = vertcat(gp.hypersamples.logL);

[quad_noise_sd, quad_input_scales] = ...
    hp_heuristics(r_X_data, r_y_data, 10);


gp = rmfield(gp,{'hyperparams','hypersamples'});
% now we completely overwrite gp

if opt.derivative_observations
    % set_gp assumes a standard homogenous covariance, we don't want to tell
    % it about derivative observations.
    plain_obs = X_data(:,end) == 0;
    
    set_X_data = X_data(plain_obs,1:end-1);
    set_y_data = y_data(plain_obs,:);
    
    
    gp = set_gp(opt.cov_fn, opt.mean_fn, gp, set_X_data, set_y_data, ...
        opt.num_hypersamples);
    
    hps_struct = set_hps_struct(gp);
    gp.covfn = @(flag) derivativise(@gp.covfn,flag);
    gp.meanfn = @(flag) wderiv_mean_fn(hps_struct,flag);
    
    gp.X_data = X_data;
    gp.y_data = y_data;
    
    
else
    gp = set_gp(opt.cov_fn, opt.mean_fn, [], X_data, y_data, ...
        opt.num_hypersamples);
end



hps_struct = set_hps_struct(gp);
input_scale_inds = hps_struct.logInputScales;
noise_sd_ind = hps_struct.logNoiseSD;
output_scale_ind = hps_struct.logOutputScale;
num_dims = size(X_data,2);
actual_log_noise_sd = log(1e-9) + gp.hyperparams(output_scale_ind).priorMean;
%gp.hyperparams(noise_sd_ind).priorMean;
big_log_noise_sd = log(0.1) + gp.hyperparams(output_scale_ind).priorMean;

gp.hyperparams(noise_sd_ind)=orderfields(...
    struct('name','logNoiseSD',...
        'priorMean',actual_log_noise_sd,...
        'priorSD',eps,...
        'NSamples',1,...
        'type','inactive'),gp.hyperparams);

sampled_gp = hyperparams(gp);
full_active_inds = sampled_gp.active_hp_inds;
hypersamples = sampled_gp.hypersamples;
num_hypersamples = numel(hypersamples);
names = {'logL', 'glogL', 'datahalf', 'datatwothirds', 'cholK', 'K', 'jitters'};
for i = 1:length(names)
    hypersamples(1).(names{i}) = nan;
end

if nargin<4
    total_num_evals = num_dims*10;
else
    total_num_evals = opt.train_evals;
end
num_input_scale_passes = max(2,ceil(total_num_evals/1e4));
num_split_evals = ...
    max(1,floor(total_num_evals/(num_input_scale_passes*(num_dims+1)+1)));
opts.maxevals = num_split_evals;

fprintf('\n Beginning retraining of GP\n');
    
tic;

parfor hypersample_ind = 1:num_hypersamples
    
    warning off
    
    hypersample = hypersamples(hypersample_ind);
    
    for num_pass = 1:num_input_scale_passes
        
        % optimise input scales
        
        hypersample.hyperparameters(noise_sd_ind) = big_log_noise_sd;

        for d = 1:length(input_scale_inds) 
            active_hp_inds = [input_scale_inds(d), noise_sd_ind];

            inputscale_hypersample = move_hypersample(...
                hypersample, gp, quad_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opts);
            hypersample.hyperparameters(input_scale_inds(d)) = ...
                inputscale_hypersample.hyperparameters(input_scale_inds(d));

        end
        
        hypersample.hyperparameters(noise_sd_ind) = actual_log_noise_sd;

         % optimise output scale

        active_hp_inds = [output_scale_ind];

        outputscale_hypersample = move_hypersample(...
                hypersample, gp, quad_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opts);
        hypersample.hyperparameters(active_hp_inds) = ...
                outputscale_hypersample.hyperparameters(active_hp_inds);
            
        fprintf('\n Pass %g:\t%g',num_pass,outputscale_hypersample.logL)

    end



    % now do a joint optimisation to finish off
    
    hypersamples(hypersample_ind) = move_hypersample(...
                hypersample, gp, quad_input_scales, ...
                full_active_inds, ...
                X_data, y_data, opts);
            
    fprintf('\n Hypersample %g:\t%g\n ',hypersample_ind,hypersamples(hypersample_ind).logL)
end

gp.hypersamples = hypersamples;
gp.X_data = X_data;
gp.y_data = y_data;
gp.active_hp_inds = full_active_inds;
gp.input_scale_inds = input_scale_inds;

fprintf('Completed retraining of GP\n')

gp.grad_hyperparams = grad_hyperparams;

r_X_data = vertcat(gp.hypersamples.hyperparameters);
r_y_data = vertcat(gp.hypersamples.logL);

[quad_noise_sd, quad_input_scales, quad_output_scale] = ...
    hp_heuristics(r_X_data, r_y_data, 10);

quad_gp.quad_noise_sd = quad_noise_sd;
quad_gp.quad_input_scales = quad_input_scales;
quad_gp.quad_output_scale = quad_output_scale;

warning on
toc;

function hypersample = move_hypersample(...
    hypersample, gp, quad_input_scales, active_hp_inds, X_data, y_data, opts)

gp.active_hp_inds = active_hp_inds;
gp.hypersamples = hypersample;
a_quad_input_scales = quad_input_scales(active_hp_inds);

flag = false;
i = 0;
while ~flag && i < opts.maxevals
    i = i+1;
    
    gp = revise_gp(X_data, y_data, gp, 'new_hps');
    a_hs=[gp.hypersamples.hyperparameters(active_hp_inds)];
    a_grad_logL = [gp.hypersamples.glogL{:}];
    a_grad_logL = a_grad_logL(active_hp_inds);
    [a_hs, flag] = simple_zoom_pt(a_hs, a_grad_logL, ...
                            a_quad_input_scales, 'maximise');
    gp.hypersamples.hyperparameters(active_hp_inds) = a_hs;
end

gp = revise_gp(X_data, y_data, gp, 'new_hps');
hypersample = gp.hypersamples;
