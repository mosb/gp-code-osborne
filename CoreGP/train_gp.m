function [gp, quad_gp] = train_gp(cov_fn, mean_fn, gp, X_data, y_data, opt)
% [gp, quad_gp] = train_gp(gp, X_data, y_data, opt)
%OR
% [gp, quad_gp] = train_gp(cov_fn, mean_fn, gp, X_data, y_data, opt);
% quad_gp contains fields quad_noise_sd, quad_input_scales and
% quad_output_scale; it describes the (quadrature) hyperparameters of the
% gp fitted to the likelihood as a function of the hyperparameters.


if ischar('cov_fn')
    % [gp, quad_gp] = train_gp(cov_fn, mean_fn, gp, X_data, y_data, opt);

    if nargin<6
        opt = struct();
    end
    
    if ~isstruct(opt)
        optim_time = opt;
        opt = struct();
        opt.optim_time = optim_time;
    end
    
    opt.cov_fn = cov_fn;
    opt.mean_fn = mean_fn;
else
    % [gp, quad_gp] = train_gp(gp, X_data, y_data, opt);
    
    gp = cov_fn;
    X_data = mean_fn;
    y_data = gp;
    if nargin<4
        opt = struct();
    else
        opt = X_data;
    end
    
    if ~isstruct(opt)
        optim_time = opt;
        opt.optim_time = optim_time;
    end
end


    
[num_data, num_dims] = size(X_data);

default_opt = struct(...
                    'cov_fn', 'sqdexp', ...
                    'mean_fn', 'constant', ...
                    'num_hypersamples', min(500, 100 * num_dims), ...
                    'derivative_observations', false, ...
                    'optim_time', 60, ...
                    'print', true, ... %if false, print no output whatsoever
                    'verbose', false, ...
                    'maxevals_hs', 10, ...
                    'plots', true, ...
                    'num_passes', 6, ...
                    'force_training', true, ...
                    'parallel', true, ...
                    'noiseless', false, ...
                    'prior_mean', 'default'); % if set to a number, that is taken as the prior mean

if ~isfield(opt,'parallel') && isfield(opt,'num_hypersamples')
    opt.parallel = opt.num_hypersamples > 1;
end
                           
names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end
opt.verbose = opt.verbose && opt.print;


if opt.print
fprintf('Beginning training of GP, budgeting for %g seconds\n', ...
    opt.optim_time);
end
start_time = cputime;

if isfield(opt, 'active_hp_inds')
    gp.active_hp_inds = opt.active_hp_inds;
end


% don't want to use likelihood gradients for BMC purposes
gp.grad_hyperparams = false;


gp = set_gp(opt.cov_fn, opt.mean_fn, gp, X_data, [], ...
    opt.num_hypersamples);
hps_struct = set_hps_struct(gp);

output_scale_ind = hps_struct.logOutputScale;
noise_ind = hps_struct.logNoiseSD;
gp.noise_ind = noise_ind;
if opt.noiseless
    gp.active_hp_inds = setdiff(gp.active_hp_inds, noise_ind);
    gp.hyperparams(noise_ind).type = 'inactive';
    gp.hyperparams(noise_ind).priorMean = ...
       gp.hyperparams(output_scale_ind).priorMean - 14; 
    
   
    % we optimise each input scale independently by also allowing the noise to
    % vary when performing each optimisation; such that the noise can soak
    % up variation due to incorrect input scales in other dimensions.
    big_noise_const = +9;
    big_noise_ind = noise_ind;
else
    big_noise_const = 0;
    big_noise_ind = noise_ind;
end
mean_inds = hps_struct.mean_inds;
if any(strcmpi(opt.prior_mean, {'optimise','optimize','train'}))
    gp.active_hp_inds = union(gp.active_hp_inds, mean_inds);
    for i = 1:length(mean_inds)
        mean_ind = mean_inds(i);
        gp.hyperparams(mean_ind).type = 'real';
    end
elseif isnumeric(opt.prior_mean)
    gp.active_hp_inds = setdiff(gp.active_hp_inds, mean_inds);
    for i = 1:length(mean_inds)
        mean_ind = mean_inds(i);
        gp.hyperparams(mean_ind).type = 'inactive';
        gp.hyperparams(mean_ind).priorMean = opt.prior_mean(i);
    end
end



if opt.derivative_observations
    % set_gp assumes a standard homogenous covariance, we don't want to tell
    % it about derivative observations.
    plain_obs = X_data(:,end) == 0;
    
    set_X_data = X_data(plain_obs,1:end-1);
    set_y_data = y_data(plain_obs,:);
    
    
    gp = set_gp(opt.cov_fn,opt.mean_fn, gp, set_X_data, set_y_data, ...
        opt.num_hypersamples);
    
    hps_struct = set_hps_struct(gp);
    gp.covfn = @(flag) derivativise(@gp.covfn,flag);
    gp.meanfn = @(flag) wderiv_mean_fn(hps_struct,flag);
    
    gp.X_data = X_data;
    gp.y_data = y_data;

    
else
    gp = set_gp(opt.cov_fn, opt.mean_fn, gp, X_data, y_data, ...
        opt.num_hypersamples);
end

% if ~isfield(gp, 'hypersamples')
% gp = create_lhs_hypersamples(gp, opt.num_hypersamples);
% end




full_active_inds = gp.active_hp_inds;


input_scale_inds = hps_struct.logInputScales;
gp.input_scale_inds = input_scale_inds;

input_inds = hps_struct.input_inds;
active_input_inds = cellfun(@(x) intersect(x, full_active_inds), ...
    input_inds, ...
    'UniformOutput', false);
active_dims = find(~cellfun(@(x) isempty(x),active_input_inds));
num_active_dims = length(active_dims);

other_active_inds = ...
    setdiff(full_active_inds, horzcat(input_inds{:})); 



num_hypersamples = numel(gp.hypersamples);
tic
gp = ...
    revise_gp(X_data, y_data, gp, 'overwrite',[], 'all', ...
    [input_scale_inds(1), noise_ind]);
hs_eval_time = toc/num_hypersamples;

hypersamples = gp.hypersamples;
gp = rmfield(gp, 'hypersamples');

r_X_data = vertcat(hypersamples.hyperparameters);
r_y_data = vertcat(hypersamples.logL);

full_active_and_noise_inds = union(full_active_inds, noise_ind);

num_hps = size(r_X_data,2);
quad_input_scales = nan(1,num_hps);
[quad_noise_sd, ...
    quad_input_scales(full_active_and_noise_inds), ...
    quad_output_scale] = ...
    hp_heuristics(r_X_data(:,full_active_and_noise_inds), r_y_data, 100);
%quad_input_scales = 10 * quad_input_scales;

% only specified in case of early return
quad_gp.quad_noise_sd = quad_noise_sd;
quad_gp.quad_input_scales = quad_input_scales;
quad_gp.quad_output_scale = quad_output_scale;

if isempty(full_active_inds)
    warning('no hyperparameters active, no training performed')
    
    gp.hypersamples = hypersamples;
    gp = revise_gp(X_data, y_data, gp, 'overwrite', []);
       
    return
end
if opt.optim_time <= 0
    warning('no time allowed for training, no training performed')
    
    gp.hypersamples = hypersamples;
    gp = revise_gp(X_data, y_data, gp, 'overwrite', []);
    
    return
end


[max_logL, max_ind] = max(r_y_data);

if opt.print
fprintf('Initial best log-likelihood: \t%g',max_logL);
end
if opt.verbose
    fprintf(', for ')
    disp_gp_hps(hypersamples, max_ind,'no_logL',...
        noise_ind, input_scale_inds, output_scale_ind, mean_inds);
end
if opt.print
fprintf('\n');
end

maxevals_hs = opt.maxevals_hs;
num_passes = opt.num_passes;

ideal_time = num_hypersamples * (...
                maxevals_hs * (num_passes * (num_dims + 1)) ...
                * hs_eval_time...
                );
% the 2 *  here is due to expected speed-up due to
% parallelisation
scale_factor = ... %2 * 
    opt.optim_time / ideal_time;

% set the allowed number of likelihood evaluations
opt.maxevals_hs = ceil(maxevals_hs * scale_factor);

if opt.maxevals_hs == 1
    warning('train_gp:insuff_time','insufficient time allowed to train GP, consider decreasing opt.num_hypersamples or increasing opt.optim_time');
    if opt.force_training
        warning('train_gp:insuff_time','proceeding with minimum possible likelihood evaluations');
        opt.maxevals_hs = 2;
    else
        gp.hypersamples = hypersamples;
        return
    end
elseif opt.verbose
    fprintf('Using %g likelihood evals per pass, per input\n', opt.maxevals_hs)
end

for num_pass = 1:num_passes
    
    if opt.verbose
        fprintf('Pass %g\n', num_pass)
    end
    
    log_input_scale_cell = cell(num_hypersamples, 1);
    input_scale_logL_cell = cell(num_hypersamples, 1);
    
    other_cell = cell(num_hypersamples,1);
    other_logL_cell = cell(num_hypersamples,1);
    
    if opt.parallel
    parfor hypersample_ind = 1:num_hypersamples
        
        warning('off','revise_gp:small_num_data');
        
        if opt.verbose
            fprintf('Hyperparameter sample %g\n',hypersample_ind)
        end
        
        big_log_noise_sd = big_noise_const + ...
            hypersamples(hypersample_ind).hyperparameters(big_noise_ind);
        actual_log_noise_sd = ...
            hypersamples(hypersample_ind).hyperparameters(noise_ind);
%         
%         
        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            big_log_noise_sd;

        % optimise input scale


        log_input_scale_cell{hypersample_ind} = cell(1, num_active_dims);
        input_scale_logL_cell{hypersample_ind} = cell(1, num_active_dims);
        for d_ind = 1:num_active_dims 
            d = active_dims(d_ind);
            active_hp_inds = [input_inds{d}, noise_ind];
            

            [inputscale_hypersample, log_input_scale_mat, input_scale_logL_mat] = ...
                move_hypersample(...
                hypersamples(hypersample_ind), gp, quad_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opt);
            
            log_input_scale_d = ...
                inputscale_hypersample.hyperparameters(input_inds{d});

            hypersamples(hypersample_ind).hyperparameters(input_inds{d}) = ...
                log_input_scale_d;
            
            log_input_scale_cell{hypersample_ind}{d_ind} = log_input_scale_mat;
            input_scale_logL_cell{hypersample_ind}{d_ind} = input_scale_logL_mat;
            
            
            if opt.verbose
                fprintf(', \t for input_scale(%g) = %g\n', ...
                    d, exp(log_input_scale_d));
            end
        end
        
        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            actual_log_noise_sd;

         % optimise other hyperparameters

        active_hp_inds = other_active_inds;

        [other_hypersample, other_mat, other_logL_mat] = ...
            move_hypersample(...
                hypersamples(hypersample_ind), gp, quad_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opt);
            
        other_cell{hypersample_ind} = other_mat;
        other_logL_cell{hypersample_ind} = other_logL_mat;
            
%         hypersamples(hypersample_ind).hyperparameters(active_hp_inds) = ...
%                 other_hypersample.hyperparameters(active_hp_inds);
            
%         if opt.verbose
%             fprintf(', \t for other = %g\n', ...
%                 exp(log_other));
%         end
% 
%         % now do a quick joint optimisation to finish off
% 
%         [hypersample] = ...
%             move_hypersample(...
%                     hypersample, gp, quad_input_scales, ...
%                     full_active_inds, ...
%                     X_data, y_data, opt);

        hypersamples(hypersample_ind) = other_hypersample;
                
        if opt.verbose
            fprintf(', \t for ');
            disp_gp_hps(hypersamples(hypersample_ind), [], 'no_logL', ...
                noise_ind, input_scale_inds, output_scale_ind, mean_inds);
            fprintf('\n');
        end

    end
    else
        for hypersample_ind = 1:num_hypersamples
        
        warning('off','revise_gp:small_num_data');
        
         if opt.verbose
            fprintf('Hyperparameter sample %g\n',hypersample_ind)
        end
        
        big_log_noise_sd = big_noise_const + ...
            hypersamples(hypersample_ind).hyperparameters(big_noise_ind);
        actual_log_noise_sd = ...
            hypersamples(hypersample_ind).hyperparameters(noise_ind);
%         
%         
        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            big_log_noise_sd;

        % optimise input scale


        log_input_scale_cell{hypersample_ind} = cell(1, num_active_dims);
        input_scale_logL_cell{hypersample_ind} = cell(1, num_active_dims);
        for d_ind = 1:num_active_dims 
            d = active_dims(d_ind);
            active_hp_inds = [input_inds{d}, noise_ind];
            

            [inputscale_hypersample, log_input_scale_mat, input_scale_logL_mat] = ...
                move_hypersample(...
                hypersamples(hypersample_ind), gp, quad_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opt);
            
            log_input_scale_d = ...
                inputscale_hypersample.hyperparameters(input_inds{d});

            hypersamples(hypersample_ind).hyperparameters(input_inds{d}) = ...
                log_input_scale_d;
            
            log_input_scale_cell{hypersample_ind}{d_ind} = log_input_scale_mat;
            input_scale_logL_cell{hypersample_ind}{d_ind} = input_scale_logL_mat;
            
            
            if opt.verbose
                fprintf(', \t for input_scale(%g) = %g\n', ...
                    d, exp(log_input_scale_d));
            end
        end
        
        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            actual_log_noise_sd;

         % optimise other hyperparameters

        active_hp_inds = other_active_inds;

        [other_hypersample, other_mat, other_logL_mat] = ...
            move_hypersample(...
                hypersamples(hypersample_ind), gp, quad_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opt);
            
        other_cell{hypersample_ind} = other_mat;
        other_logL_cell{hypersample_ind} = other_logL_mat;
            
%         hypersamples(hypersample_ind).hyperparameters(active_hp_inds) = ...
%                 other_hypersample.hyperparameters(active_hp_inds);
            
%         if opt.verbose
%             fprintf(', \t for other = %g\n', ...
%                 exp(log_other));
%         end
% 
%         % now do a quick joint optimisation to finish off
% 
%         [hypersample] = ...
%             move_hypersample(...
%                     hypersample, gp, quad_input_scales, ...
%                     full_active_inds, ...
%                     X_data, y_data, opt);

        hypersamples(hypersample_ind) = other_hypersample;
                
        if opt.verbose
            fprintf(', \t for ');
            disp_gp_hps(hypersamples(hypersample_ind), [], 'no_logL', ...
                noise_ind, input_scale_inds, output_scale_ind, mean_inds);
            fprintf('\n');
        end

        end
    end
    
    % now we estimate the scale of variation of the likelihood wrt
    % log input_scale.
    
    log_input_scale_compcell = cat(1,log_input_scale_cell{:});
    input_scale_logL_compcell = cat(1,input_scale_logL_cell{:});
    quad_noise_sds = nan(num_dims+1,1);
    quad_output_scales = nan(num_dims+1,1);
        
    for d_ind = 1:num_active_dims
        a_hps_mat = cat(1, log_input_scale_compcell{:,d_ind});
        logL_mat = cat(1, input_scale_logL_compcell{:,d_ind});
        
        sorted_logL_mat = sort(logL_mat);
              
        top_inds = logL_mat >= sorted_logL_mat(round(0.9*end));
        a_hps_mat = a_hps_mat(top_inds,:);
        logL_mat = logL_mat(top_inds,:);

        [quad_noise_sds(d_ind), a_quad_input_scales, quad_output_scales(d_ind)] = ...
            hp_heuristics(a_hps_mat,logL_mat,10);

        quad_input_scales(input_scale_inds(d_ind)) = a_quad_input_scales(1);
    end
    
    % now we estimate the scale of variation of the likelihood wrt
    % log other and log noise sd.
    a_hps_mat = cat(1,other_cell{:});
    logL_mat = cat(1,other_logL_cell{:});
    
    a_hps_mat = max(a_hps_mat, -100);
    logL_mat = max(logL_mat,-1e100);
    sorted_logL_mat = sort(logL_mat);

    top_inds = logL_mat >= sorted_logL_mat(round(0.9*end));
    a_hps_mat = a_hps_mat(top_inds,:);
    logL_mat = logL_mat(top_inds,:);

    [quad_noise_sds(end), a_quad_input_scales, quad_output_scales(end)] = ...
        hp_heuristics(a_hps_mat,logL_mat,10);

    quad_input_scales(other_active_inds) = a_quad_input_scales;
    
    quad_noise_sd = min(quad_noise_sds);
    quad_output_scale = max(quad_output_scales);
    
    if (cputime-start_time > opt.optim_time) && num_pass > 0 ...
        || num_pass == num_passes
        % need to end here so that we have gp that has been trained on the
        % whole dataset, rather than on a `close' subset
        break
    end  
  
    if opt.print
    fprintf('\n');
    end
    
end

gp.hypersamples = hypersamples;
gp.X_data = X_data;
gp.y_data = y_data;
gp.active_hp_inds = full_active_inds;
gp.input_scale_inds = input_scale_inds;
gp.output_scale_ind = output_scale_ind;

[max_logL, max_ind] = max([gp.hypersamples.logL]);

if opt.print
fprintf('Final best log-likelihood: \t%g',max_logL);
end
if opt.verbose
    fprintf(', for ')
    disp_gp_hps(gp, max_ind, 'no_logL');
elseif opt.print
    fprintf('\n');
end


quad_gp.quad_noise_sd = (quad_noise_sd);
quad_gp.quad_input_scales = (quad_input_scales); 
quad_gp.quad_output_scale = (quad_output_scale);

if opt.print
    fprintf('Completed retraining of GP in %g seconds\n', cputime-start_time)
    fprintf('\n');
end


function [hypersample, a_hps_mat, logL_mat] = move_hypersample(...
    hypersample, gp, quad_input_scales, active_hp_inds, X_data, y_data, opt)

gp.hypersamples = hypersample;
a_quad_input_scales = quad_input_scales(active_hp_inds);

flag = false;
i = 0;
a_hps_mat = nan(opt.maxevals_hs,length(active_hp_inds));
logL_mat = nan(opt.maxevals_hs,1);

if opt.verbose && opt.plots
    for a = 1:length(active_hp_inds)
        figure(a);clf; hold on;
        title(['Optimising hyperperparameter ',num2str(a)])
    end
end

broken = false;

while (~flag || ceil(opt.maxevals_hs/5) > i) && i < opt.maxevals_hs-1
    i = i+1;
    
    try
        gp = ...
            revise_gp(X_data, y_data, gp, 'overwrite', [], 'all', active_hp_inds);
    catch
        broken = true;
        i = i - 1;
        break;
    end
    
    logL = gp.hypersamples.logL;
    a_hs=gp.hypersamples.hyperparameters(active_hp_inds);
    
    a_hps_mat(i,:) = a_hs;
    logL_mat(i) = logL;
    
    if opt.verbose && opt.plots
        for a = 1:length(active_hp_inds)
            figure(a)
            x = a_hs(a);
            plot(x, logL, '.');

            g = [gp.hypersamples.glogL{a}];
            scale = a_quad_input_scales(a);

            line([x-scale,x+scale],...
                [logL-g*scale,logL+g*scale],...
                'Color',[0 0 0],'LineWidth',1.5);
        end
    end
    
    if i>1 && logL_mat(i) < backup_logL
        % the input scale which predicted the largest increase in logL is
        % likely wrong
        
        dist_moved = (a_hs - backup_a_hs).*a_grad_logL;
        [dummy,max_ind] = max(dist_moved);

        a_quad_input_scales(max_ind) = 0.5*a_quad_input_scales(max_ind);
        
%         [~,a_quad_input_scales] = ...
%             hp_heuristics(a_hps_mat(1:i,:),logL_mat(1:i,:),10);
%         
        a_hs = backup_a_hs;
    else
        backup_logL = logL;
        backup_a_hs = a_hs;
        a_grad_logL = [gp.hypersamples.glogL{active_hp_inds}];
    end
    

    [a_hs, flag] = simple_zoom_pt(a_hs, a_grad_logL, ...
                            a_quad_input_scales, 'maximise');
    gp.hypersamples.hyperparameters(active_hp_inds) = a_hs;
    
end

if ~broken
    try

    
    gp = revise_gp(X_data, y_data, gp, 'overwrite');
    logL = gp.hypersamples.logL;
    a_hs = gp.hypersamples.hyperparameters(active_hp_inds);

    i = i+1;
    
    a_hps_mat(i,:) = a_hs;
    logL_mat(i) = logL;
    catch
    end
end

a_hps_mat = a_hps_mat(1:i,:);
logL_mat = logL_mat(1:i,:);

[max_logL,max_ind] = max(logL_mat);
gp.hypersamples.hyperparameters(active_hp_inds) = a_hps_mat(max_ind,:);
gp = revise_gp(X_data, y_data, gp, 'overwrite');
hypersample = gp.hypersamples;

% not_nan = all(~isnan([a_hps_mat,logL_mat]),2);
% 
% [quad_noise_sd, a_quad_input_scales] = ...
%     hp_heuristics(a_hps_mat(not_nan,:), logL_mat(not_nan), 10);
% quad_input_scales(active_hp_inds) = a_quad_input_scales;

if opt.verbose
fprintf('LogL: %g -> %g',logL_mat(1), max_logL)
end
if opt.verbose && opt.plots
    %keyboard;
end

% hp = 4;
% log_ins = linspace(-5,10, 1000);
% logLs = nan(1000,1);
% gp.hypersamples = hypersample;
% for i =1:1000;
%     gp.hypersamples(1).hyperparameters(hp) = log_ins(i);
%     gp = revise_gp(X_data, y_data, gp, 'overwrite', [], 'all', hp);
%     logLs(i) = gp.hypersamples(1).logL;
%     dlogLs(i) = gp.hypersamples(1).glogL{hp};
% end
% clf
% hold on
% plot(log_ins, (logLs),'r')
% plot(log_ins, (dlogLs),'k')

    