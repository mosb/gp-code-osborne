function gp = set_gp(covfn_name, meanfn_name, gp, X_data, y_data, num_hypersamples)
% gp = set_gp(covfn_name, meanfn_name, gp, X_data, y_data,
% num_hypersamples)
% covfn_name can be 'sqdexp', 'matern', 'ratquad', 'poly' or 'prodcompact'
% (all of which can be read about in Rasmussen & Williams) and meanfn_name
% can be 'constant', 'planar' or 'quadratic'. The mean function's
% hyperparameters are all set by performing a least-squares fit.

if nargin<6
    num_hypersamples = 1000;
end
num_hypersamples = max(1, ceil(num_hypersamples));

if isempty(gp) 
    gp = struct();
end
if ~isfield(gp,'hyperparams')
    num_existing_samples = 1;
    num_existing_hps = 0;
    
    gp.hyperparams(1) = ...
        struct('name','dummy',...
        'priorMean',nan,...
        'priorSD',nan,...
        'NSamples',nan,...
        'type',nan);
    
else

    default_vals = struct('priorSD', 1, ...
                        'NSamples', 1, ...
                        'type', 'inactive');

    names = fieldnames(default_vals);
    for i = 1:length(names);
        name = names{i};
        for ind = 1:numel(gp.hyperparams)
            if (~isfield(gp.hyperparams(ind), name)) ...
                    || isempty(gp.hyperparams(ind).(name))
                gp.hyperparams(ind).(name) = default_vals.(name);
            end
        end
    end

    num_existing_samples = prod([gp.hyperparams.NSamples]);
    num_existing_hps = numel(gp.hyperparams);
end
hps_struct = set_hps_struct(gp);


have_X_data = nargin >= 4 && ~isempty(X_data);
have_y_data = nargin >= 5 && ~isempty(y_data);
create_logNoiseSD = ~isfield(hps_struct,'logNoiseSD') ...
                        && ~isfield(gp,'noisefn');
create_logInputScales = ~isfield(hps_struct,'logInputScales') ...
                        || isempty(hps_struct.logInputScales);
create_logOutputScale = ~isfield(hps_struct,'logOutputScale')...
                        || isempty(hps_struct.logOutputScale);
create_covfn = ~isempty(covfn_name) ||...
                        ~isfield(gp,'covfn')...
                        || isempty(gp.covfn);
create_meanfn = ~isempty(meanfn_name) ||...
                        (~isfield(gp,'meanfn')...
                        || isempty(gp.meanfn)) && ...
                (~isfield(gp,'meanPos')...
                        || isempty(gp.meanPos));
                  



if have_X_data
    num_dims = size(X_data,2);
    num_hps_to_create = ...
        create_logNoiseSD + num_dims*create_logInputScales + create_logOutputScale;

    num_samples = factor_in_odds(num_hypersamples/num_existing_samples,num_hps_to_create);
    if size(X_data,1) == 1
        input_scales = X_data;
        input_SD = 2;
    else
        input_scales = std(X_data);
        input_SD = 2;
    end
end

if have_y_data
    output_scale = std(y_data);
    output_SD = 1;
else
    output_scale = exp(10);
    output_SD = 3;
end

if create_logNoiseSD
%     if have_data
    noise_ind = incr_num_hps(gp);
    gp.logNoiseSDPos = noise_ind;

    gp.hyperparams(noise_ind) = orderfields(...
        struct('name','logNoiseSD',...
        'priorMean',log(0.1*output_scale),...
        'priorSD',output_SD,...
        'NSamples',num_samples(noise_ind-num_existing_hps),...
        'type','real'),...
        gp.hyperparams);
%     else
%         disp('Need to specify a prior for logNoiseSD, or include data to create one')
%     end
end
if create_logInputScales
    if have_X_data
        inputs_ind = nan(1,num_dims);
        for dim = 1:num_dims   
            inputs_ind(dim) = incr_num_hps(gp);
            
            gp.hyperparams(inputs_ind(dim)) = orderfields(...
                struct('name',['logInputScale',num2str(dim)],...
                'priorMean',log(input_scales(dim)),...
                'priorSD',input_SD,...
                'NSamples',num_samples(inputs_ind(dim)-num_existing_hps),...
                'type','real'),gp.hyperparams);
        end
        gp.input_scale_inds = inputs_ind;
    else
        disp('Need to specify a prior for logInputScales, or include data to create one')
    end
else
    gp.input_scale_inds = hps_struct.logInputScales;  
end
if create_logOutputScale
%     if have_data
        output_ind = incr_num_hps(gp);
        
        gp.hyperparams(output_ind) = orderfields(...
            struct('name','logOutputScale',...
            'priorMean',log(output_scale),...
            'priorSD',output_SD,...
            'NSamples',num_samples(output_ind-num_existing_hps),...
            'type','real'),...
            gp.hyperparams);
%     else
%         disp('Need to specify a prior for logOutputScale, or include data to create one')
%     end
end

if create_meanfn
    if have_y_data
        switch meanfn_name
            case 'constant'
                gp = set_constant_mean(gp, X_data, y_data);
            case 'affine'
                gp = set_affine_mean(gp, X_data, y_data);
            case 'quadratic'
                gp = set_quadratic_mean(gp, X_data, y_data);
            otherwise
                % assume constant.
                gp = set_constant_mean(gp, X_data, y_data);
        end
    else
        gp = set_constant_mean(gp, 1);
    end
end

hps_struct = set_hps_struct(gp);
if create_covfn
    % set the covariance function
    gp.covfn = @(flag) hom_cov_fn(hps_struct,covfn_name,flag);
    if ~isfield(gp, 'sqd_diffs_cov')
        gp.sqd_diffs_cov = true;
    end
elseif nargin(gp.covfn) == 2
    % we have not initialised the cov fn with hps_struct yet
    gp.covfn = @(flag) gp.covfn(hps_struct,flag);
end

if have_y_data && have_X_data
    
    gp = set_gp_data(gp, X_data, y_data);

    Mu = get_mu(gp);
    
    y_data_minus_mu = y_data - Mu([gp.hyperparams.priorMean]', X_data);
    
    [est_noise_sd,est_input_scales,est_output_scale] = ...
        hp_heuristics(X_data,y_data_minus_mu);
    

    if create_logNoiseSD
        gp.hyperparams(noise_ind).priorMean = log(est_noise_sd);
        gp.hyperparams(noise_ind).priorSD = 0.5;
    end
    if create_logInputScales
        for dim = 1:num_dims  
            gp.hyperparams(inputs_ind(dim)).priorMean = ...
                log(est_input_scales(dim));
            gp.hyperparams(inputs_ind(dim)).priorSD = 1.5;            
        end
    end
    if create_logOutputScale
        gp.hyperparams(output_ind).priorMean = log(est_output_scale);
        gp.hyperparams(output_ind).priorSD = 1.5;
    end
end

function num = incr_num_hps(gp)
if ~isfield(gp,'hyperparams') || ...
        strcmpi(gp.hyperparams(1).name,'dummy')
    num = 1;
else
    num = numel(gp.hyperparams)+1;
end

function factors = factor_in_odds(big_number,num_factors)
big_number = floor(big_number);
num_factors = max(1,num_factors);

if big_number<3^num_factors
    number_of_threes=floor(log(big_number)/log(3));
    factors = ones(num_factors,1);
    factors(2:(1+number_of_threes)) = 3;
else
    odd_num = floor(big_number^(1/(num_factors)));
    if odd_num/2 == floor(odd_num/2)
        odd_num = odd_num-1;
    end
    factors = ones(num_factors,1) * odd_num;
    ind = 1; % this is deliberate, we want to add samples to input scales before the noise
    while prod(factors)<big_number
        ind = ind+1;
        if ind > num_factors
            ind = 1;
        end
        factors(ind) = odd_num;
        if ind == 1
            odd_num = odd_num+2;
        end
    end
    factors(ind) = factors(ind)-2;
end
