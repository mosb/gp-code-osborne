function noise_fun = get_diag_noise(gp, flag)
% usgae:
% diag_noise = get_diag_noise(gp, 'plain');
% if grad_hyperparams
%     Ddiag_noise = get_diag_noise(gp, {'grad hyperparams',grad_hp_inds});
% end

if nargin<2
    flag = 'plain';
end

noise_fun_test = isfield(gp,'noisefn');
if ~noise_fun_test 
    if (~isfield(gp, 'logNoiseSDPos'))
        names = {gp.hyperparams.name};
        logNoiseSDPos = cellfun(@(x) strcmpi(x, 'logNoiseSD'), names);
    else
        logNoiseSDPos = gp.logNoiseSDPos;
    end
    
    % if logNoiseSDPos is empty, this means we assume zero noise.
    if ~isempty(logNoiseSDPos)
        noise_fun = diag_iid_noise_fn(logNoiseSDPos, flag);
    else
        noise_fun = @(X,hps) 0; % no matter the flag
    end
else
    noise_fun = gp.noisefn(flag);
end

function noise_mat = diag_iid_noise_fn(logNoiseSDPos, flag)

if nargin<2
    flag='plain';
end

[flag, grad_hp_inds] = process_flag(flag);
% no effort to compute gradient wrt hyperparams, so do it for all anyway
switch flag
    case 'plain'
        noise_mat = @(hps, Xs) exp(hps(logNoiseSDPos))*ones(size(Xs,1));
    case 'grad inputs'
        noise_mat=@(hps, Xs) Dinputs_noise_mat(Xs);
    case 'hessian inputs'
        noise_mat=@(hps, Xs) Dinputs_noise_mat(Xs);
    case 'grad hyperparams'
        noise_mat=@(hps, Xs) Dhps_noise_mat(hps, Xs, logNoiseSDPos);
end

function Dnoise_mat = Dinputs_noise_mat(Xs)

[L,num_dims] = size(Xs);
Dnoise_mat = mat2cell2d(zeros(num_dims*L,L),L*ones(num_dims,1),L);

function Dnoise_mat = Dhps_noise_mat(hps, Xs, logNoiseSDPos)

L = size(Xs,1);
num_hps = length(hps);

Dnoise_mat = mat2cell2d(zeros(num_hps*L,L),L*ones(num_hps,1),L);
Dnoise_mat{logNoiseSDPos} = 2 * (e