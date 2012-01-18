function [xpc_unc] = expected_uncertainty_evidence...
      (hs_a, sample_struct, prior_struct, r_gp, opt)
% returns the expected negative-squared-mean-evidence after adding a
% hyperparameter sample hs_a. This quantity is a scaled version of the
% expected variance in the evidence.
%
% [exp_log_unc] = ...
%   expected_uncertainty_evidence(hs_a, sample_struct, prior_struct, r_gp, widths_quad_input_scales, opt)
% - hs_a (1 by the number of hyperparameters) is a row vector expressing
% the relevant trial hyperparameters. If hs_a is empty or omitted, then the
% returned mean and variance are the non-expected qantities for the
% evidence.
% - evidence: the current evidence
% - sample_struct requires fields
% * samples
% * log_r
% - prior_struct requires fields
% * means
% * sds
% - (optional) input r_gp has fields
% * quad_output_scale
% * quad_noise_sd
% * quad_input_scales
% * hs_c
% * R_s
% * yot_s
% * Yot_sc_s
%
% alternatively:
% [exp_log_unc] = 
%    expected_uncertainty_evidence(hs_a, gp, [], r_gp, opt)
% - gp requires fields:
% * hyperparams(i).priorMean
% * hyperparams(i).priorSD
% * hypersamples.logL
%
% - output r_gp has the same fields as input r_gp plus
                        
no_r_gp = nargin<4;
if nargin<5
    opt = struct();
end

default_opt = struct('gamma_const', 100, ...
                    'allowed_cond_error',10^-14, ...
                    'sds_tr_input_scales', false);
% sds_tr_input_scales represents the posterior standard deviations in the
% input scales for tr. If false, a delta function posterior is assumed.
             
names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end

if ~isempty(prior_struct)
    % evidence(hs_a, sample_struct, prior_struct, r_gp, opt)
    
    hs_s = sample_struct.samples;
    log_r_s = sample_struct.log_r;
    
    [num_s, num_hps] = size(hs_s);
    
    prior_means = prior_struct.means;
    prior_sds = prior_struct.sds;
    
else
    % evidence(hs_a, gp, [], r_gp, opt)
    gp = sample_struct;
    
    hs_s = vertcat(gp.hypersamples.hyperparameters);
    log_r_s = vertcat(gp.hypersamples.logL);
    
    [num_s, num_hps] = size(hs_s);
    
    prior_means = vertcat(gp.hyperparams.priorMean);
    prior_sds = vertcat(gp.hyperparams.priorSD);
    
end

hs_c = r_gp.hs_c;
hs_sc = [hs_s; hs_c];
num_sc = size(hs_sc, 1);

hs_sa = [hs_s; hs_a];
num_sa = num_s + 1;

prior_var = prior_sds.^2;
prior_var_stack = reshape(prior_var, 1, 1, num_hps);

[max_log_r_s, max_ind] = max(log_r_s);
% this function is only ever used to compare different hs_a's for the
% single fixed r_s, so no big deal about subtracting off this
log_r_s = log_r_s - max_log_r_s;
r_s = exp(log_r_s);


tilde = @(x, gamma_x) log(bsxfun(@rdivide, x, gamma_x) + 1);
%inv_tilda = @(tx, gamma_x) exp(bsxfun(@plus, tx, log(gamma_x))) - gamma_x;

gamma = opt.gamma_const;
tr_s = tilde(r_s, gamma);


% r is assumed to have zero mean
if no_r_gp || isempty(r_gp)
    [r_noise_sd, r_input_scales, r_output_scale] = ...
        hp_heuristics(hs_s, r_s, 10);

    r_sqd_output_scale = r_output_scale^2;
else
    r_sqd_output_scale = r_gp.quad_output_scale^2;
    r_input_scales = r_gp.quad_input_scales;
end


r_sqd_lambda = r_sqd_output_scale* ...
    prod(2*pi*r_input_scales.^2)^(-0.5);

del_input_scales = 0.5 * r_input_scales;
del_sqd_output_scale = r_sqd_output_scale;

sqd_r_input_scales = r_input_scales.^2;
sqd_r_input_scales_stack = reshape(sqd_r_input_scales,1,1,num_hps);
sqd_del_input_scales_stack = reshape(del_input_scales.^2,1,1,num_hps);

hs_a_minus_mean = hs_a - prior_means';

hs_sc_minus_mean_stack = reshape(bsxfun(@minus, hs_sc, prior_means'),...
                    num_sc, 1, num_hps);
sqd_hs_sc_minus_mean_stack = ...
    repmat(hs_sc_minus_mean_stack.^2, 1, 1, 1);

hs_a_minus_mean_stack = reshape(hs_a_minus_mean,...
                    1, 1, num_hps);
sqd_hs_a_minus_mean_stack = ...
    repmat(hs_a_minus_mean_stack.^2, num_sc, 1, 1);

sqd_dist_stack_sc_a = reshape(bsxfun(@minus, hs_sc, hs_a).^2, ...
                    num_sc, 1, num_hps);
sqd_dist_stack_sa_a = reshape(bsxfun(@minus, hs_sa, hs_a).^2, ...
                    num_sa, 1, num_hps);

K_s = r_gp.K_s;
R_s = r_gp.R_s;

% we update the covariance matrix over r. We consider all old jitters fixed
% and add in jitter at hs_a sufficient to render the matrix
% well-conditioned (maybe we should revisit old jitters, even if it'd be
% slower?).
K_sa = nan(num_sa);
diag_K_sa = diag_inds(K_sa);
K_sa(diag_K_sa(1:num_s)) = diag(K_s);
K_sa_a = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_sa_a, sqd_r_input_scales_stack), 3));
K_sa(num_sa, :) = K_sa_a;
K_sa(:, num_sa) = K_sa_a';

% this importances vector is to force the jitter to be applied solely to
% the added point hs_a. improve_covariance_conditioning will automatically
% do this so long as K_sa has nans in the appropriate off-diagonal
% elements, but not if K_sa is 2x2, so that there are no off-diagonal
% elements.
importances = [inf(num_s,1);0];
K_sa = improve_covariance_conditioning(K_sa, ...
    importances, opt.allowed_cond_error);
        
R_sa = updatechol(K_sa, R_s, num_sa);

sum_prior_var_sqd_input_scales_r = ...
    prior_var + sqd_r_input_scales;

yot_a = r_sqd_output_scale * ...
    prod(2*pi*sum_prior_var_sqd_input_scales_r)^(-0.5) * ...
    exp(-0.5 * ...
    sum(hs_a_minus_mean.^2./sum_prior_var_sqd_input_scales_r));

yot_s = r_gp.yot_s;
yot_sa = [yot_s; yot_a];

yot_inv_K_sa = solve_chol(R_sa, yot_sa)';


               

                
prior_var_times_sqd_dist_stack_sc_a = bsxfun(@times, prior_var_stack, ...
                    sqd_dist_stack_sc_a);
                
opposite_del = sqd_del_input_scales_stack;
opposite_r = sqd_r_input_scales_stack;
sqd_r_input_scales_stack = reshape(r_input_scales.^2,1,1,num_hps);
sqd_del_input_scales_stack = reshape(del_input_scales.^2,1,1,num_hps);
inv_determ_del_r = (prior_var_stack.*(...
        sqd_r_input_scales_stack + sqd_del_input_scales_stack) + ...
        sqd_r_input_scales_stack.*sqd_del_input_scales_stack).^(-1);
% 2 pi is outside of sqrt because each element of determ is actually the
% determinant of a 2 x 2 matrix
Yot_sc_a = del_sqd_output_scale * r_sqd_output_scale * ...
    prod(1/(2*pi) * sqrt(inv_determ_del_r)) .* ...
    exp(-0.5 * sum(bsxfun(@times,inv_determ_del_r,...
                bsxfun(@times, opposite_r, ...
                    sqd_hs_sc_minus_mean_stack) ...
                + bsxfun(@times, opposite_del, ...
                    sqd_hs_a_minus_mean_stack) ...
                + prior_var_times_sqd_dist_stack_sc_a...
                ),3));
            
%     % some code to test that this construction works          
%     Lambda = diag(prior_sds.^2);
%     W_del = diag(del_input_scales.^2);
%     W_r = diag(r_input_scales.^2);
%     mat = kron(ones(2),Lambda)+blkdiag(W_del,W_r);
% 
%     Yot_sc_a_test = @(i) del_sqd_output_scale * r_sqd_output_scale *...
%         mvnpdf([hs_sc(i,:)';hs_a'],[prior_means';prior_means'],mat);

Yot_sc_s = r_gp.Yot_sc_s;
Yot_sc_sa = [Yot_sc_s, Yot_sc_a];

del_inv_K = r_gp.del_inv_K;
del_inv_K_Yot_inv_K_sa = del_inv_K * solve_chol(R_sa, Yot_sc_sa')';

n_sa = del_inv_K_Yot_inv_K_sa + yot_inv_K_sa;

lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

sqd_dist_s_a = bsxfun(@minus, hs_s, hs_a).^2;  
K_s_a = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_s_a, sqd_r_input_scales), 2));

invR_K_s_a = linsolve(R_s, K_s_a, lowr);                
K_invK_a_s = linsolve(R_s, invR_K_s_a, uppr)';

n_a = n_sa(num_sa);
% zero prior mean
tm_a = K_invK_a_s * tr_s;

tv_a = r_sqd_lambda - invR_K_s_a' * invR_K_s_a;

if opt.sds_tr_input_scales
    
    % we correct for the impact of learning r_t on our belief about the
    % input scales
    
    C = opt.sds_tr_input_scales.^2;
    if size(C,1) == 1
        C = C';
    end
    
    invK_tr_s = solve_chol(R_s, tr_s);

    sqd_dist_stack_s = r_gp.sqd_dist_stack_s;
    sqd_dist_stack_s_a = reshape(sqd_dist_s_a', 1, num_s, num_hps);

    %each plate is the derivative with respect to a different log input
    %scale
    
    DK_a_s = bsxfun(@times, K_s_a', ...
        bsxfun(@rdivide, ...
        sqd_dist_stack_s_a', ...
        sqd_r_input_scales_stack));
    DK_s = bsxfun(@times, K_s, ...
        bsxfun(@rdivide, ...
        sqd_dist_stack_s, ...
        sqd_r_input_scales_stack));

    Dtm_a = prod3(DK_a_s, invK_tr_s) ...
            - prod3(K_invK_a_s, prod3(DK_s, invK_tr_s));
        
    tv_a = tv_a + sum(bsxfun(@times, Dtm_a.^2, C));
end


n_r_s = n_sa(1:num_s) * r_s;

xpc_unc =  - n_r_s^2 ...
    - 2 * n_r_s * (gamma * exp(tm_a + 0.5*tv_a) - gamma) ...
    - n_a^2 * gamma^2 * ...
        (exp(2*tm_a + 2*tv_a) - 2 * exp(tm_a + 0.5*tv_a) + 1);