function hps_struct = set_hps_struct(covvy,num_dims)
% Creates a structure hps_struct containing the positions of each
% hyperparameter

if isfield(covvy,'hyperparams')
    num_hps = numel(covvy.hyperparams);

    names = {covvy.hyperparams.name};
    posns_vec = 1:num_hps;
    posns = num2cell(posns_vec);

    joint = [names;posns];
    hps_struct = struct(joint{:});
    
    hps_struct.num_hps = num_hps;

    is_input_scale_cell = strfind(names,'logInputScale');
    input_scale_inds = find(~cellfun(@(x) isempty(x),is_input_scale_cell));

    if ~isempty(input_scale_inds)
        hps_struct.logInputScales = input_scale_inds;
    end
    
%         is_planar_weight_cell = strfind(names,'PlanarMeanWeight');
%     planar_weight_inds = find(~cellfun(@(x) isempty(x),is_planar_weight_cell));
% 
%     hps_struct.PlanarMeanWeights = planar_weight_inds;
% 
%     is_quad_weight_cell = strfind(names,'QuadMeanWeight');
%     quad_weight_inds = find(~cellfun(@(x) isempty(x),is_quad_weight_cell));
% 
%     hps_struct.QuadMeanWeights = quad_weight_inds;
%     
    is_multiple_hp = strfind(names,'1');
    multiple_hp_inds = find(~cellfun(@(x) isempty(x),is_multiple_hp));
    
    for ind = 1:length(multiple_hp_inds)
        hp = multiple_hp_inds(ind);
        name = names{hp};
        name = name(1:(is_multiple_hp{hp}-1));
        is_hp = strfind(names,name);
        hp_inds = find(~cellfun(@(x) isempty(x),is_hp));
        hps_struct.([name,'s']) = hp_inds;
    end
    
    if nargin<2 && isfield(hps_struct,'logInputScales')
        num_dims = length(hps_struct.logInputScales);
    elseif nargin<2 && isfield(hps_struct,'log_w0s')
        num_dims = length(hps_struct.log_w0s);
    elseif  nargin<2 && isfield(covvy,'num_dims')
        num_dims = covvy.num_dims;
    elseif nargin<2 && ~isfield(covvy,'num_dims')
        num_dims = 'unknown';
    end
    hps_struct.num_dims = num_dims;
else
    hps_struct = struct([]);
end