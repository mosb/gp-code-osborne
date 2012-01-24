function [log_mean_evidence, r_gp] = ...
    log_evidence(sample_struct, prior_struct, r_gp, opt)
% Returns the log-mean-evidence, and a structure r_gp to ease its
% future computation.
%
% [log_mean_evidence, r_gp] = ...
% evidence(sample_struct, prior_struct, r_gp, opt)
% - sample_struct requires fields
%     * samples
%     * log_r
% - prior_struct requires fields
%     * means
%     * sds
% - (optional) input r_gp has fields
%     * quad_output_scale
%     * quad_noise_sd
%     * quad_input_scales
%
% alternatively:
% [log_mean_evidence, r_gp] = evidence(gp, [], r_gp, opt)
% - gp requires fields:
%     * hyperparams(i).priorMean
%     * hyperparams(i).priorSD
%     * hypersamples.logL
%
% - output r_gp has the same fields as input r_gp plus
%     * hs_c
%     * R_s
%     * yot_s
%     * Yot_sc_s

no_r_gp = nargin<3;
if nargin<4
    opt = struct();
end

default_opt = struct('num_c', 100,...
                    'gamma_const', 100, ...
                    'num_box_scales', 5, ...
                    'allowed_cond_error',10^-14, ...
                    'update', false, ...
                    'print', true);
                
names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end

% If we have actually only added a single new sample at position opt.update,
% can do efficient sequential updates.
% updating = isnumeric(opt.update);

if ~isempty(prior_struct)
    % evidence(hs_t, sample_struct, prior_struct, r_gp, opt)
    
    hs_s = sample_struct.samples;
    log_r_s = sample_struct.log_r;
    
    [num_s, num_hps] = size(hs_s);
    
    prior_means = prior_struct.means;
    prior_sds = prior_struct.sds;
    
else
    % evidence(hs_t, gp, [], r_gp, opt)
    gp = sample_struct;
    
    hs_s = vertcat(gp.hypersamples.hyperparameters);
    log_r_s = vertcat(gp.hypersamples.logL);
    
    [num_s, num_hps] = size(hs_s);
    
    prior_means = vertcat(gp.hyperparams.priorMean);
    prior_sds = vertcat(gp.hyperparams.priorSD);
    
end


opt.num_c = min(opt.num_c, num_s);
num_c = opt.num_c;

prior_sds_stack = reshape(prior_sds, 1, 1, num_hps);
prior_var_stack = prior_sds_stack.^2;

[max_log_r_s, max_ind] = max(log_r_s);
log_r_s = log_r_s - max_log_r_s;
r_s = exp(log_r_s);


% r is assumed to have zero mean
if no_r_gp || isempty(r_gp)
    [r_noise_sd, r_input_scales, r_output_scale] = ...
        hp_heuristics(hs_s, r_s, 10);

    r_sqd_output_scale = r_output_scale^2;
    
    r_gp = struct();
else
    r_sqd_output_scale = r_gp.quad_output_scale^2;
    r_input_scales = r_gp.quad_input_scales;
end

% we force GPs for r and tr to share hyperparameters. del_rr are assumed to
% all have input scales equal to half of those for r.

min_input_scales = r_input_scales;

r_sqd_lambda = r_sqd_output_scale* ...
    prod(2*pi*r_input_scales.^2)^(-0.5);

del_input_scales = 0.5 * r_input_scales;
del_sqd_output_scale = r_sqd_output_scale;
del_sqd_lambda = del_sqd_output_scale* ...
    prod(2*pi*del_input_scales.^2)^(-0.5);

lower_bound = min(hs_s) - opt.num_box_scales*min_input_scales;
upper_bound = max(hs_s) + opt.num_box_scales*min_input_scales;

% find the candidate points, far removed from existing samples
try
    hs_c = find_farthest(hs_s, [lower_bound; upper_bound], num_c, ...
                         min_input_scales);
catch
    warning('find_farthest failed')
    hs_c = far_pts(hs_s, [lower_bound; upper_bound], num_c);
end

hs_sc = [hs_s; hs_c];
num_sc = size(hs_sc, 1);
num_c = num_sc - num_s;

sqd_dist_stack_sc = bsxfun(@minus,...
                reshape(hs_sc,num_sc,1,num_hps),...
                reshape(hs_sc,1,num_sc,num_hps))...
                .^2;  

    
sqd_dist_stack_s = sqd_dist_stack_sc(1:num_s, 1:num_s, :);

sqd_r_input_scales_stack = reshape(r_input_scales.^2,1,1,num_hps);
sqd_del_input_scales_stack = reshape(del_input_scales.^2,1,1,num_hps);
                
K_r_s = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s, sqd_r_input_scales_stack), 3)); 
[K_r_s, jitters_r_s] = improve_covariance_conditioning(K_r_s, ...
    r_s, ...
    opt.allowed_cond_error);
R_r_s = chol(K_r_s);

K_del_sc = del_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_sc, sqd_del_input_scales_stack), 3)); 
importance_sc = ones(num_sc,1);
importance_sc(num_s + 1 : end) = 2;
K_del_sc = improve_covariance_conditioning(K_del_sc, importance_sc, ...
    opt.allowed_cond_error);
R_del_sc = chol(K_del_sc);     

sqd_dist_stack_s_sc = sqd_dist_stack_sc(1:num_s, :, :);

K_r_s_sc = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s_sc, sqd_r_input_scales_stack), 3));   
       
sum_prior_var_sqd_input_scales_stack_r = ...
    prior_var_stack + sqd_r_input_scales_stack;
sum_prior_var_sqd_input_scales_stack_del = ...
    prior_var_stack + sqd_del_input_scales_stack;

opposite_del = sqd_del_input_scales_stack;
opposite_r = sqd_r_input_scales_stack;
    
hs_sc_minus_mean_stack = reshape(bsxfun(@minus, hs_sc, prior_means'),...
                    num_sc, 1, num_hps);
sqd_hs_sc_minus_mean_stack = ...
    repmat(hs_sc_minus_mean_stack.^2, [1, num_sc, 1]);
tr_sqd_hs_sc_minus_mean_stack = tr(sqd_hs_sc_minus_mean_stack);

yot_r_s = r_sqd_output_scale * ...
    prod(2*pi*sum_prior_var_sqd_input_scales_stack_r)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, hs_sc_minus_mean_stack(1:num_s, :, :).^2, ...
    sum_prior_var_sqd_input_scales_stack_r),3));

yot_inv_K_r = solve_chol(R_r_s, yot_r_s)';

yot_del_sc = del_sqd_output_scale * ...
    prod(2*pi*sum_prior_var_sqd_input_scales_stack_del)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, hs_sc_minus_mean_stack(:, :, :).^2, ...
    sum_prior_var_sqd_input_scales_stack_del),3));

yot_inv_K_del = solve_chol(R_del_sc, yot_del_sc)';

prior_var_times_sqd_dist_stack_sc = bsxfun(@times, prior_var_stack, ...
                    sqd_dist_stack_sc);
                
% 2 pi is outside of sqrt because each element of determ is actually the
% determinant of a 2 x 2 matrix

inv_determ_del_r = (prior_var_stack.*(...
        sqd_r_input_scales_stack + sqd_del_input_scales_stack) + ...
        sqd_r_input_scales_stack.*sqd_del_input_scales_stack).^(-1);
Yot_del_r = del_sqd_output_scale * r_sqd_output_scale * ...
    prod(1/(2*pi) * sqrt(inv_determ_del_r)) .* ...
    exp(-0.5 * sum(bsxfun(@times,inv_determ_del_r,...
                bsxfun(@times, opposite_r, ...
                    sqd_hs_sc_minus_mean_stack(1:num_sc, 1:num_s, :)) ...
                + bsxfun(@times, opposite_del, ...
                    tr_sqd_hs_sc_minus_mean_stack(1:num_sc, 1:num_s, :)) ...
                + prior_var_times_sqd_dist_stack_sc(1:num_sc, 1:num_s, :)...
                ),3));
            
Yot_inv_K_del_r = solve_chol(R_r_s, Yot_del_r')';
            
% some code to test that this construction works          
% Lambda = diag(prior_sds.^2);
% W_qd = diag(qd_input_scales.^2);
% W_r = diag(r_input_scales.^2);
% mat = kron(ones(2),Lambda)+blkdiag(W_qd,W_r);
% 
% Yot_qd_r_test = @(i,j) qd_sqd_output_scale * r_sqd_output_scale *...
%     mvnpdf([hs_s(i,:)';hs_s(j,:)'],[prior_means';prior_means'],mat);
            

% As = [hs_sc_minus_mean_stack(3,:);hs_sc_minus_mean_stack(4,:)];
% Bs = 0*[prior_means';prior_means'];
% scalesss = [input_scales;del_input_scales].^2;
% covmat = kron2d(diag(prior_sds.^2), ones(2)) + diag(scalesss(:));
% sqd_output_scale.^2 * mvnpdf(As(:),Bs(:),covmat);


tilde = @(x, gamma_x) log(bsxfun(@rdivide, x, gamma_x) + 1);
%inv_tilda = @(tx, gamma_x) exp(bsxfun(@plus, tx, log(gamma_x))) - gamma_x;

% this is after r_s has already been divided by exp(max_log_r_s). tr_s is
% its correct value, but log(gamma) has effectively had max_log_r_s
% subtracted from it
gamma_r = opt.gamma_const;
tr_s = tilde(r_s, gamma_r);

lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

two_thirds_r = linsolve(R_r_s,linsolve(R_r_s, K_r_s_sc, lowr), uppr)';
mean_r_sc =  two_thirds_r * r_s;
mean_tr_sc = two_thirds_r * tr_s;

% use a crude thresholding here as our tilde transformation will fail if
% the mean goes below zero
mean_r_sc = max(mean_r_sc, eps);
Delta_tr_sc = mean_tr_sc - tilde(mean_r_sc, gamma_r);

del_inv_K = solve_chol(R_del_sc, Delta_tr_sc)';


minty_r = yot_inv_K_r * r_s;
minty_del_r = del_inv_K * Yot_inv_K_del_r * r_s;
minty_del = yot_inv_K_del * Delta_tr_sc;

mean_ev = minty_r + minty_del_r + gamma_r * minty_del;

log_mean_evidence = max_log_r_s + log(mean_ev);


r_gp.hs_c = hs_c;
r_gp.sqd_dist_stack_s = sqd_dist_stack_s;

r_gp.R_r_s = R_r_s;
r_gp.K_r_s = K_r_s;
r_gp.yot_r_s = yot_r_s;

r_gp.R_del_sc = R_del_sc;
r_gp.K_del_sc = K_del_sc;
r_gp.yot_del_sc = yot_del_sc;

r_gp.Delta_tr_sc = Delta_tr_sc;

r_gp.Yot_sc_s = Yot_del_r;
