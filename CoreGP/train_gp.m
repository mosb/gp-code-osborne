function [gp] = train_gp(covfn_name, meanfn_name, gp, XData, yData, ...
    opt)
% train gp

warning off

if nargin<6
    opt = struct();
elseif ~isstruct(opt)
    opt.optim_time = opt;
end

default_opt = struct('print', true, ...
                        'optim_time', 300);

names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end

min_opt.print = opt.print;
min_opt.showits = 0;

gp = set_gp(covfn_name, meanfn_name, gp, XData, yData, 1);
gp = hyperparams(gp);
full_active_inds = gp.active_hp_inds;
hps_struct = set_hps_struct(gp);
input_scale_inds = hps_struct.logInputScales;
noise_sd_ind = hps_struct.logNoiseSD;
output_scale_ind = hps_struct.logOutputScale;
num_dims = length(input_scale_inds);

actual_log_noise_sd = gp.hyperparams(noise_sd_ind).priorMean;
big_log_noise_sd = log(0.1) + gp.hyperparams(output_scale_ind).priorMean;


tic;
neg_log_likelihood...
    (full_active_inds,gp.hypersamples(1).hyperparameters(full_active_inds),...
    gp, XData, yData, min_opt);
time_for_one_likelihood = toc;
total_num_evals = floor(opt.optim_time/time_for_one_likelihood);

num_input_scale_passes = max(3,ceil(total_num_evals/1e4));

if opt.print
 fprintf('Beginning optimisation of hyperparameters; time budget: %g seconds, budgeting for %d evaluations.\n', ...
        opt.optim_time, total_num_evals);
  % of course, this doesn't account for parfor savings, or for the
  % overheads of direct itself.
end
    
tic;

num_split_evals = ...
    max(1,floor(total_num_evals/(num_input_scale_passes*(num_dims+1)+1)));

min_opt.maxevals = num_split_evals;

% switch derivatives off now.
if isfield(gp, 'grad_hyperparams')
    grad_hyperparams = gp.grad_hyperparams;
else
    grad_hyperparams = false;
end
gp.grad_hyperparams = false;


matlabpool close force
matlabpool open

for num_pass = 1:num_input_scale_passes
    
    if opt.print
    fprintf('Beginning optimisation of input scales, pass %g of %g\n', num_pass, num_input_scale_passes)
    end  
    
    inputscales_priorMean = nan(1,num_dims);
    inputscales_priorSD = nan(1,num_dims);

    gp.hypersamples(1).hyperparameters(noise_sd_ind) = big_log_noise_sd;
 

    parfor d = 1:num_dims
        
        active_hp_inds = [input_scale_inds(d), noise_sd_ind];

        [best_a_hypersample,est_sds] = minimise_active_inds(gp, active_hp_inds, XData, yData, min_opt);

        inputscales_priorMean(d) = best_a_hypersample(1);
        inputscales_priorSD(d) = est_sds(1);
        
    end
    
    gp.hyperparameters(noise_sd_ind).hyperparameters(noise_sd_ind) = actual_log_noise_sd;

    for d = 1:num_dims
        gp.hyperparams(input_scale_inds(d)).priorMean = inputscales_priorMean(d);
        gp.hyperparams(input_scale_inds(d)).priorSD = inputscales_priorSD(d);
    end
 
    gp.hypersamples(1).hyperparameters(input_scale_inds) = inputscales_priorMean;
    %gp.hypersamples(1).hyperparameters(full_active_inds)'
    
    if opt.print
    fprintf('Beginning optimisation of output scale and noise, pass %g of %g\n', num_pass, num_input_scale_passes)
    end
    % optimise output scale & noise

    active_hp_inds = [output_scale_ind, noise_sd_ind];

    [best_a_hypersample,est_sds] = minimise_active_inds(gp, active_hp_inds, XData, yData, min_opt);

    gp.hyperparams(output_scale_ind).priorMean = best_a_hypersample(1);
    gp.hyperparams(output_scale_ind).priorSD = est_sds(1);
    gp.hyperparams(noise_sd_ind).priorMean = best_a_hypersample(2);
    gp.hyperparams(noise_sd_ind).priorSD = est_sds(2);
    gp.hypersamples(1).hyperparameters(active_hp_inds) = best_a_hypersample;

    gp.hypersamples(1).hyperparameters(full_active_inds)'
end



% now do a joint optimisation to finish off

if opt.print
fprintf('Beginning final, joint optimisation\n')
end

[best_a_hypersample,est_sds,neg_logL] = minimise_active_inds(gp, full_active_inds, XData, yData, min_opt);

gp.hypersamples(1).hyperparameters(full_active_inds) = best_a_hypersample;
gp.hypersamples(1).logL = -neg_logL;

if opt.print
fprintf('Completed optimisation of hyperparameters\n')
gp.hypersamples(1).hyperparameters(full_active_inds)'

fprintf('Maximum likelihood hyperparameters:\n');
cellfun(@(name, value) fprintf('\t%s\t%g\n', name, value), ...
    {gp.hyperparams.name}', ...
    mat2cell2d(gp.hypersamples(1).hyperparameters',...
    ones(numel(gp.hyperparams),1),1));
end

gp.grad_hyperparams = grad_hyperparams;
gp = revise_gp(XData, yData, gp, 'overwrite');

matlabpool close
warning on

toc;

function [neg_logL neg_glogL] = neg_log_likelihood...
    (active_hp_inds,a_hypersample, gp, XData, yData, min_opt)

want_derivs = nargout>1;
gp.grad_hyperparams = want_derivs;

if size(a_hypersample,1)>1
    gp.hypersamples(1).hyperparameters(active_hp_inds) = a_hypersample';
else
    gp.hypersamples(1).hyperparameters(active_hp_inds) = a_hypersample;
end

gp = revise_gp(XData, yData, gp, 'overwrite');

neg_logL = - gp.hypersamples(1).logL;
if want_derivs
    neg_glogL = - [gp.hypersamples(1).glogL{:}];
end
if min_opt.print
fprintf('.');
end



function [best_a_hypersample, est_sds, neg_logL] = minimise_active_inds(gp,...
    active_hp_inds, XData, yData, min_opt)


a_priorMeans=[gp.hyperparams(active_hp_inds).priorMean];
a_priorSDs=[gp.hyperparams(active_hp_inds).priorSD];
a_lower_bound = a_priorMeans - 3*a_priorSDs;
a_upper_bound = a_priorMeans + 3*a_priorSDs;


objective = @(a_hypersample) neg_log_likelihood...
    (active_hp_inds,a_hypersample, gp, XData, yData, min_opt);

Problem.f = objective;
a_bounds = [a_lower_bound; a_upper_bound]';

% N = 10;
% grid_pts = allcombs([linspace(a_lower_bound(1),a_upper_bound(1),N)',...
%     linspace(a_lower_bound(2),a_upper_bound(2),N)']);
% obj = nan(length(grid_pts),1);
% for i=1:length(grid_pts)
%     obj(i) = objective(grid_pts(i,:));
% end
% inp = reshape(grid_pts(:,1),N,N);
% nois = reshape(grid_pts(:,2),N,N);
% negloglik = reshape(obj,N,N);
% contourf(inp,nois,negloglik);
% xlabel('log input scale')
% ylabel('log noise sd')
% [min_negloglik,min_ind] = min(obj);
% best_a_hypersample = grid_pts(min_ind,:);
% est_sds = 1/3*a_priorSDs;

opt.showits = min_opt.showits;
opt.maxevals = min_opt.maxevals;

[neg_logL, best_a_hypersample, history] = Direct(Problem, a_bounds, opt);

best_a_hypersample = best_a_hypersample';

ratio = 1;
% if size(history,1)>=2
%     diff_history = diff(history(:,3));
%     ratio = diff_history(end)/min(diff_history);
% else
%     ratio = 1;
% end
est_sds =  max(ratio, 1/3) * a_priorSDs;
if min_opt.print
fprintf('\n logL = %g\n', -neg_logL)
end
