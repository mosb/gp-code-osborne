function [mean_out, sd_out, unadj_mean_out, unadj_sd_out] = ...
    predict_bq(sample_struct, priol_struct, ...
    l_gp_hypers_SE, tl_gp_hypers_SE, del_gp_hypers_SE, ...
    qd_gp_hypers_SE, qdd_gp_hypers_SE, opt)
% [mean_out, sd_out, unadj_mean_out, unadj_sd_out] = ...
%    predict_bq(sample_struct, priol_struct, ...
%    l_gp_hypers_SE, tl_gp_hypers_SE, ...
%    qd_gp_hypers_SE, qdd_gp_hypers_SE, opt)
% 
% return the posterior mean and sd by marginalising hyperparameters.
%
% OUTPUTS
% - mean_out
% - sd_out
% - unadj_mean_out: mean without correction for delta
% - unadj_sd_out: sd without correction for delta
%
% INPUTS
% - samples: requires fields
%   - locations
%   - scaled_l: likelihoods divided by maximum likelihood
%   - tl: log-transformed scaled likelihoods
%   - max_log_l: max log likelihood
%   - qd: (alternatively: mean_y, such that mean_y = qd) predictive mean
%   - qdd: (alternatively: var_y, such that var_y = qdd - qd^2) predictive
%       second moment
% - prior requires fields
%   - mean
%   - covariance
% - l_gp_hypers_SE: hypers for sqd exp covariance over l, with fields
%   - log_output_scale
%   - log_input_scales
% - tl_gp_hypers_SE: hypers for sqd exp covariance over tl, with fields
%   - log_output_scale
%   - log_input_scales
% - del_gp_hypers_SE: hypers for sqd exp covariance over del, with fields
%   - log_output_scale
%   - log_input_scales
% - qd_gp_hypers_SE: hypers for sqd exp covariance over l, with fields
%   - log_output_scale
%   - log_input_scales
%   - prior_mean
% - qdd_gp_hypers_SE: hypers for sqd exp covariance over tl, with fields
%   - log_output_scale
%   - log_input_scales
%   - prior_mean
% - ev_params: (see log_evidence.m) has fields
%   - x_c
%   - sqd_dist_stack_s
%   - R_tl_s
%   - K_tl_s
%   - inv_K_tl_s
%   - jitters_l
%   - sqd_dist_stack_s
%   - R_del
%   - K_del
%   - ups_l
%   - ups_del
%   - Ups_sc_s
%   - del_inv_K_del
%   - delta_tl_sc
%   - minty_del
%   - log_mean_second_moment

% Load options, set to default if not available
% ======================================================

if nargin<6
    opt = struct();
end

default_opt = struct('num_c', 400,...
                    'gamma_const', 1, ...
                    'num_box_scales', 5, ...
                    'no_adjustment', false, ...
                    'allowed_cond_error',10^-14, ... % allowed conditioning error
                    'print', true);
                
opt = set_defaults( opt, default_opt );


% Load data
% ======================================================
    
x_s = samples.locations;
num_s = size(x_s);

% rescale all log-likelihood values for numerical accuracy; we'll correct
% for this at the end of the function
l_s = samples.scaled_l;

% opt.gamma is corrected for after l_s has already been divided by
% exp(samples.max_log_l_s). tl_s is its correct value, but log(opt.gamma) has
% effectively had samples.max_log_l_s subtracted from it. 
tl_s = samples.tl;

% candidate locations
x_c = ev_params.x_c;


if isfield(sample_struct, 'mean_y')

    mean_y = sample_struct.mean_y;
    var_y = sample_struct.var_y;

    % these quantities need to be num_s by num_star matrices
    if size(mean_y, 1) ~= num_s
        mean_y = mean_y';
    end
    if size(var_y, 1) ~= num_s
        var_y = var_y';
    end

    qd_s = mean_y;
    qdd_s = var_y + mean_y.^2;

elseif isfield(sample_struct, 'qd') 

    qd_s = sample_struct.qd;
    if isfield(sample_struct, 'qdd')
        qdd_s = sample_struct.qdd;
    else
        qdd_s = sample_struct.qd;
    end

end
num_star = size(qd_s, 2);
    

% rescale by subtracting appropriate prior means
qdmm_s = bsxfun(@minus, qd_s, qd_gp_hypers_SE.prior_mean);
qddmm_s = bsxfun(@minus, qdd_s, qdd_gp_hypers_SE.prior_mean);

tqdd_s = log_transform(qd_s);

% IMPORTANT NOTE: THIS NEEDS TO BE CHANGED TO MATCH WHATEVER MEAN IS USED
% FOR qdd
mu_tqdd = tqdd_s(max_ind,:);
tqddmm_s = bsxfun(@minus, tqdd_s, mu_tqdd);


% Compute our covariance matrices and their cholesky factors
% ======================================================

% input hyperparameters are for a sqd exp covariance, whereas in all that
% follows we use a gaussian covariance. We correct the output scales
% appropriately.
l_gp_hypers = sqdexp2gaussian(l_gp_hypers_SE);
tl_gp_hypers = sqdexp2gaussian(tl_gp_hypers_SE);
qd_gp_hypers = sqdexp2gaussian(qd_gp_hypers_SE);
qdd_gp_hypers = sqdexp2gaussian(qdd_gp_hypers_SE);
del_gp_hypers = sqdexp2gaussian(del_gp_hypers_SE);

% we assume the gps for qd and tqd share hyperparameters, as do qdd and
% tqdd. eps_rr, eps_qdr, eps_rqdd, eps_qddr are assumed to all have the
% same hypers as del (the output scales are probably wildly wrong, but are
% not that important as they should cancel out)
tqd_gp_hypers = qd_gp_hypers;
tqdd_gp_hypers = qdd_gp_hypers;

sqd_dist_stack_s = ev_params.sqd_dist_stack_s;
sqd_dist_stack_s_sc = ev_params.sqd_dist_stack_s_sc;

importance_s = l_s.*mean(abs(qdmm_s),2);

% The gram matrix over the predictive mean qd and its cholesky factor
K_qd = gaussian_mat(sqd_dist_stack_s, qd_gp_hypers);
K_qd = improve_covariance_conditioning(K_qd, ...
    importance_s, ...
    opt.allowed_cond_error);
R_qd = chol(K_qd);
inv_K_qd_qdmm = solve_chol(R_qd, qdmm_s);
% The covariance over the transformed qdd between x_sc and x_s
K_qd_sc = gaussian_mat(sqd_dist_stack_s_sc, tqd_gp_hypers);

% The gram matrix over the predictive second moment qdd and its cholesky 
% factor
K_qdd = gaussian_mat(sqd_dist_stack_s, qdd_gp_hypers);
K_qdd = improve_covariance_conditioning(K_qdd, ...
    importance_s, ...
    opt.allowed_cond_error);
R_qdd = chol(K_qdd);
inv_K_qdd_qddmm = solve_chol(R_qdd, qddmm_s);
% The covariance over the transformed qdd between x_sc and x_s
K_qdd_sc = gaussian_mat(sqd_dist_stack_s_sc, tqdd_gp_hypers);

% The gram matrix over the transformed predictive second moment qdd and its
% cholesky factor
K_tqdd = gaussian_mat(sqd_dist_stack_s, tqdd_gp_hypers);
K_tqdd = improve_covariance_conditioning(K_tqdd, ...
    importance_s, ...
    opt.allowed_cond_error);
R_tqdd = chol(K_tqdd);
inv_K_tqdd_tqddmm = solve_chol(R_tqdd, tqddmm_s);
% The covariance over the transformed qdd between x_sc and x_s
K_tqdd_sc = gaussian_mat(sqd_dist_stack_s_sc, tqdd_gp_hypers);

% The gram matrix over the predictive second moment qdd and its cholesky 
% factor
R_eps = ev_params.R_del_sc;
 
% Compute eps quantities
% ======================================================

% the mean of qd at x_sc
mean_qd_sc =  bsxfun(@plus, mu_qd, K_qd_sc' * inv_K_qd_qdmm);

% the mean of qdd at x_sc
mean_qdd_sc =  bsxfun(@plus, mu_qdd, K_qdd_sc' * inv_K_qdd_qddmm);
% use a crude thresholding here as our tilde transformation will fail if
% the mean goes below zero
mean_qdd_sc = max(mean_qdd_sc, eps);

% the mean of the transformed (log) likelihood at x_sc
mean_tqdd_sc = bsxfun(@plus, mu_tqdd, K_tqdd_sc' * inv_K_tqdd_tqddmm);

% the difference between the mean of the transformed (log) likelihood and
% the transform of the mean likelihood
delta_tqdd_sc = mean_tqdd_sc - log_transform(mean_qdd_sc, opt.gamma);

mean_l_sc = ev_params.mean_tl_sc;
delta_tl_sc = ev_params.delta_tl_sc;

% Compute eps quantities, the scaled difference between the mean of the
% transformed (log) quantities and the transform of the mean quantities
eps_ll_sc = mean_l_sc .* delta_tl_sc;
eps_qdl_sc = bsxfun(@times, mean_qd_sc, delta_tl_sc);
eps_lqdd_sc = bsxfun(@times, mean_l_sc, delta_tqdd_sc);
eps_qddl_sc = bsxfun(@times, mean_qdd_sc, delta_tl_sc);

% Compute various Gaussian-derived quantities required to evaluate the mean
% integrals over qd and qdd
% ======================================================
       

ups_inv_K_eps

inv_K_Yot_inv_K_qd_l

inv_K_Yot_inv_K_qd_eps

    inv_K_Yot_inv_K_qdd_r
    
    inv_K_Yot_inv_K_qdd_eps

minty_r = ev_params.minty_l;
minty_delta_tl = ups_inv_K_eps * delta_tl_sc;
minty_eps_rr = ups_inv_K_eps * eps_ll_sc;
minty_eps_qdr = (ups_inv_K_eps * eps_qdl_sc)';
minty_eps_rqdd = (ups_inv_K_eps * eps_lqdd_sc)';
minty_eps_qddr = (ups_inv_K_eps * eps_qddl_sc)';




% all the quantities below need to be adjusted to account for the non-zero
% prior means of qd and qdd
minty_qd_r = qdmm_s' * inv_K_Yot_inv_K_qd_l * l_s;
rhod = minty_qd_r / minty_r + mu_qd';
minty_qd_eps_rr = qdmm_s' * inv_K_Yot_inv_K_qd_eps * eps_ll_sc + ...
                mu_qd' * minty_eps_rr;

             
minty_qdd_r = qddmm_s' * inv_K_Yot_inv_K_qdd_r * l_s;
rhodd = minty_qdd_r / minty_r + mu_qdd';
% only need the diagonals of this quantity, the full covariance is not
% required
minty_qdd_eps_rqdd = ...
    sum((qddmm_s' * inv_K_Yot_inv_K_qdd_eps) .* eps_lqdd_sc', 2) + ...
                mu_qdd' .* minty_eps_rqdd;
minty_qdd_eps_rr = qddmm_s' * inv_K_Yot_inv_K_qdd_eps * eps_ll_sc + ...
                mu_qdd' * minty_eps_rr;





adj_rhod_tr = (minty_qd_eps_rr + gamma_r * minty_eps_qdr ...
                -(minty_eps_rr + gamma_r * minty_delta_tl) * rhod) / minty_r;         
adj_rhodd_tq = (minty_qdd_eps_rqdd + gamma_qdd * minty_eps_rqdd) / minty_r;
adj_rhodd_tr = (minty_qdd_eps_rr + gamma_r * minty_eps_qddr ...
    -(minty_eps_rr + gamma_r * minty_delta_tl) * rhodd) / minty_r;
end
if opt.no_adjustment
    adj_rhod_tr = 0;
    adj_rhodd_tq = 0;
    adj_rhodd_tr = 0;
end


mean_out = rhod + adj_rhod_tr;
unadj_mean_out = mean_out;
second_moment = rhodd + adj_rhodd_tq + adj_rhodd_tr;
if want_posterior
    mean_out = second_moment;
    sd_out = nan;
else
    var_out = second_moment - mean_out.^2;
    problems = var_out<0;
    var_out(problems) = qdd_s(max_ind,problems) - qd_s(max_ind,problems).^2;
    
    sd_out = sqrt(var_out);
    
    var_out = rhodd - mean_out.^2;
    problems = var_out<0;
    var_out(problems) = qdd_s(max_ind,problems) - qd_s(max_ind,problems).^2;
    
    unadj_sd_out = sqrt(var_out);
end

% for i = 1:num_hps
%     figure(i);clf;
%     hold on
%     plot(phi(:,i), qd, '.k');
%     plot(phi(:,i), qdd, '.r');
%     xlabel(['x_',num2str(i)]);
% end
% 
% [qd_noise_sd, qd_input_scales, qd_output_scale] = ...
%         hp_heuristics(phi, qd, 10);
%     
% [qdd_noise_sd, qdd_input_scales, qdd_output_scale] = ...
%         hp_heuristics(phi, qdd, 10);