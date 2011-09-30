function [noise, output_scale, input_scales] = disp_gp_hps(gp, best_ind, flag)

if ~isfield(gp, 'hypersamples') && isfield(gp, 'logL')
    gp1.hypersamples = gp;
    gp = gp1;
end

if nargin<3
    flag= [];
end

if nargin<2 || isempty(best_ind)
    if isfield(gp.hypersamples, 'logL')
        [logL, best_ind] = max([gp.hypersamples.logL]);
    else
        best_ind = 1;
    end
end
    
best_hypersample = gp.hypersamples(best_ind).hyperparameters;
num_hps = length(best_hypersample);

    
if isfield(gp, 'input_scale_inds')
    noise_ind = gp.noise_ind;
    output_scale_ind = gp.output_scale_ind;
    input_scale_inds = gp.input_scale_inds;
else
    noise_ind = 1;
    input_scale_inds = 2:(num_dims+1);
    output_scale_ind = num_dims+2;
end

noise = exp(best_hypersample(noise_ind));
output_scale = exp(best_hypersample(output_scale_ind));
input_scales = exp(best_hypersample(input_scale_inds));


    
if nargout == 0
    if ~strcmpi(flag,'no_logL')
        fprintf('log-likelihood of %g for ', logL);
    end
    fprintf('hyperparameters:\n');
    
    fprintf('noise SD:\t%g\n', ...
        noise);
    fprintf('output scale:\t%g\n', ...
        output_scale*(prod(2*pi*input_scales))^(-1/4));
    fprintf('input_scales:\t[');
    fprintf(' %f', input_scales);
    fprintf(']\n');
    
end
    
