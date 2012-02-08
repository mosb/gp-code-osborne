function [log_mean_evidence, log_var_evidence, r_gp_params] = ...
    log_evidence(samples, prior, r_gp_params, opt)
% Returns the log-mean-evidence, and a structure r_gp_params to ease its
% future computation.
%
% [log_mean_evidence, r_gp_params] = ...
% evidence(sample_struct, prior_struct, r_gp_params, opt)
% - samples requires fields
%     * samples
%     * log_r
% - prior requires fields
%     * means
%     * sds
% - (optional) input r_gp_params has fields
%     * quad_output_scale
%     * quad_noise_sd
%     * quad_input_scales
%
%
% - output r_gp_params has the same fields as input r_gp_params plus
%     * hs_c
%     * R_s
%     * ups_s
%     * Ups_sc_s

no_r_gp_params = nargin<3;
if nargin<4
    opt = struct();
end

default_opt = struct('num_c', 100,... % number of candidate points
                    'gamma_const', (exp(1)-1)^(-1), ... % numerical scaling factor
                    'num_box_scales', 5, ... % defines the box over which to take candidates
                    'allowed_cond_error',10^-14, ... % allowed conditioning error
                    'update', false); % update log_evidence with a single new point
opt = set_defaults( opt, default_opt );

% Note: If we have actually only added a single new sample at position opt.update,
% can do efficient sequential updates.
% updating = isnumeric(opt.update);
% for now, we recompute evidence from scratch each time
[num_samples, num_sample_dims] = size(samples.locations);

% create stacks of the standard deviations and variances of the priors for
% each hyperparameter
prior_var = diag(prior.covariance)';
prior_sds = sqrt(prior_var);
prior_sds_stack = reshape(prior_sds, 1, 1, num_sample_dims);
prior_var_stack = reshape(prior_var, 1, 1, num_sample_dims);

% rescale all log-likelihood values for numerical accuracy; we'll correct
% for this at the end of the function
max_log_r_s = max(samples.log_r);
r_s = exp(samples.log_r - max_log_r_s);

% hyperparameters for gp over the log-likelihood, r, assumed to have zero
% mean
if no_r_gp_params || isempty(r_gp_params)
    [r_noise_sd, r_input_scales, r_output_scale] = ...
        hp_heuristics(samples.locations, r_s, 10);

    r_sqd_output_scale = r_output_scale^2;
    
    r_gp_params = struct();
else
    r_sqd_output_scale = r_gp_params.quad_output_scale^2;
    r_input_scales = r_gp_params.quad_input_scales;
end

% hyperparameters for gp over delta, the difference between log-gp-mean-r and
% gp-mean-log-r
min_input_scales = r_input_scales;

r_sqd_lambda = r_sqd_output_scale* ...
    prod(2*pi*r_input_scales.^2)^(-0.5);

del_input_scales = 0.5 * r_input_scales;
del_sqd_output_scale = r_sqd_output_scale;
del_sqd_lambda = del_sqd_output_scale* ...
    prod(2*pi*del_input_scales.^2)^(-0.5);

lower_bound = min(samples.locations) - 2*min_input_scales;
lower_bound = max(lower_bound, prior.mean - opt.num_box_scales*prior_sds);

upper_bound = max(samples.locations) + 2*min_input_scales;
upper_bound = min(upper_bound, prior.mean + opt.num_box_scales*prior_sds);

opt.num_c = min(opt.num_c, num_samples);
num_c = opt.num_c;

% find the candidate points, far removed from existing samples
try
    candidate_locations = find_farthest(samples.locations, [lower_bound; upper_bound], num_c, ...
                         min_input_scales);
catch
    warning('find_farthest failed')
    candidate_locations = far_pts(samples.locations, [lower_bound; upper_bound], num_c);
end

sample_locs_combined = [samples.locations; candidate_locations];
num_samples_combined = size(sample_locs_combined, 1);

sqd_dist_stack_sc = bsxfun(@minus,...
                reshape(sample_locs_combined,num_samples_combined,1,num_sample_dims),...
                reshape(sample_locs_combined,1,num_samples_combined,num_sample_dims))...
                .^2;  

    
sqd_dist_stack_s = sqd_dist_stack_sc(1:num_samples, 1:num_samples, :);

r_input_scales_stack = reshape(r_input_scales,1,1,num_sample_dims);
sqd_r_input_scales_stack = r_input_scales_stack.^2;
sqd_del_input_scales_stack = reshape(del_input_scales.^2,1,1,num_sample_dims);
                
K_r_s = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s, sqd_r_input_scales_stack), 3)); 
[K_r_s, jitters_r_s] = improve_covariance_conditioning(K_r_s, ...
    r_s, ...
    opt.allowed_cond_error);
R_r_s = chol(K_r_s);
inv_K_r_r_s = solve_chol(R_r_s, r_s);

K_del_sc = del_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_sc, sqd_del_input_scales_stack), 3)); 
importance_sc = ones(num_samples_combined,1);
importance_sc(num_samples + 1 : end) = 2;
K_del_sc = improve_covariance_conditioning(K_del_sc, importance_sc, ...
    opt.allowed_cond_error);
R_del_sc = chol(K_del_sc);     

sqd_dist_stack_s_sc = sqd_dist_stack_sc(1:num_samples, :, :);


K_r_s_sc = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s_sc, sqd_r_input_scales_stack), 3));   
       
ups_var_stack_r = ...
    prior_var_stack + sqd_r_input_scales_stack;
ups_var_stack_del = ...
    prior_var_stack + sqd_del_input_scales_stack;
% detvar terms define the diagonal elements of the covariance C used in the
% appropriate multiplicative factors (det 2 pi C)^(-0.5)
ups2_detvar_stack_r = ...
    2 * prior_var_stack + sqd_r_input_scales_stack;
ups2_var_stack_r = ...
    prior_var_stack + 2 * sqd_r_input_scales_stack ...
     - ups_var_stack_r .* ...
    (2 * prior_var_stack + sqd_r_input_scales_stack).^(-1) ...
        .*  ups_var_stack_r;
chi_detvar_stack = 2 * prior_var_stack + sqd_r_input_scales_stack;
chi3_detvar_stack_r = ...
    2 * prior_var_stack + sqd_r_input_scales_stack ...
    - 2 * prior_var_stack .* ...
        ups_var_stack_r.^(-1) .*  prior_var_stack;
chi3_var_stack_r = prior_var_stack .* ...
    (3 * prior_var_stack .* r_input_scales_stack +...
     3 * prior_sds_stack .* sqd_r_input_scales_stack +...
     r_input_scales_stack.^3);

opposite_del = sqd_del_input_scales_stack;
opposite_r = sqd_r_input_scales_stack;
    
hs_sc_minus_mean_stack = reshape(bsxfun(@minus, sample_locs_combined, prior.mean),...
                    num_samples_combined, 1, num_sample_dims);
sqd_hs_sc_minus_mean_stack = ...
    repmat(hs_sc_minus_mean_stack.^2, [1, num_samples_combined, 1]);
tr_sqd_hs_sc_minus_mean_stack = tr(sqd_hs_sc_minus_mean_stack);

% ups_s = int K(hs, hs_s)  prior(hs) dhs
ups_r_s = r_sqd_output_scale * ...
    prod(2*pi*ups_var_stack_r)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, hs_sc_minus_mean_stack(1:num_samples, :, :).^2, ...
    ups_var_stack_r),3));
ups_inv_K_r = solve_chol(R_r_s, ups_r_s)';

ups_del_sc = del_sqd_output_scale * ...
    prod(2*pi*ups_var_stack_del)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, hs_sc_minus_mean_stack(:, :, :).^2, ...
    ups_var_stack_del),3));
ups_inv_K_del = solve_chol(R_del_sc, ups_del_sc)';

% ups2_s = int int K(hs, hs') K(hs', hs_s) prior(hs) prior(hs') dhs dhs'
ups2_r_s = r_sqd_output_scale^2 * ...
    prod(2*pi*ups2_detvar_stack_r)^(-0.5) * ...
    prod(2*pi*ups2_var_stack_r)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, hs_sc_minus_mean_stack(1:num_samples, :, :).^2, ...
    ups2_var_stack_r),3));
ups2_inv_K_r = solve_chol(R_r_s, ups2_r_s)';     

% chi = int int K(hs, hs') prior(hs) prior(hs') dhs dhs'
chi_r = r_sqd_output_scale * ...
    prod(2*pi*chi_detvar_stack)^(-0.5);
                
% chi3_s_s = int int K(hs_s, hs) K(hs, hs') K(hs', hs_s) prior(hs)
% prior(hs') dhs dhs'
% NB: we do not have to include the prod(2*pi*chi3_var_stack_r)^(-0.5) term
chi3_r = r_sqd_output_scale * ...
    prod(2*pi*chi3_detvar_stack_r)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, sqd_dist_stack_s, ...
    chi3_var_stack_r),3));
chi3_r = chi3_r .* (ups_r_s * ups_r_s');
r_inv_K_chi3_inv_K_r = inv_K_r_r_s' * chi3_r * inv_K_r_r_s;

prior_var_times_sqd_dist_stack_sc = bsxfun(@times, prior_var_stack, ...
                    sqd_dist_stack_sc);
                
% Ups_s_s = int K(hs_s, hs) K(hs, hs_s) prior(hs) dhs
inv_determ_del_r = (prior_var_stack.*(...
        sqd_r_input_scales_stack + sqd_del_input_scales_stack) + ...
        sqd_r_input_scales_stack.*sqd_del_input_scales_stack).^(-1);
Ups_del_r = del_sqd_output_scale * r_sqd_output_scale * ...
    prod(1/(2*pi) * sqrt(inv_determ_del_r)) .* ...
    exp(-0.5 * sum(bsxfun(@times,inv_determ_del_r,...
                bsxfun(@times, opposite_r, ...
                    sqd_hs_sc_minus_mean_stack(1:num_samples_combined, 1:num_samples, :)) ...
                + bsxfun(@times, opposite_del, ...
                    tr_sqd_hs_sc_minus_mean_stack(1:num_samples_combined, 1:num_samples, :)) ...
                + prior_var_times_sqd_dist_stack_sc(1:num_samples_combined, 1:num_samples, :)...
                ),3));
% 2 pi is outside of sqrt because each element of determ is actually the
% determinant of a 2 x 2 matrix
Ups_inv_K_del_r = solve_chol(R_r_s, Ups_del_r')';


inv_determ_r_r = (2*prior_var_stack.*sqd_r_input_scales_stack + ...
        sqd_r_input_scales_stack.*sqd_r_input_scales_stack).^(-1);
Ups_r_r = r_sqd_output_scale^2 * ...
    prod(1/(2*pi) * sqrt(inv_determ_r_r)) .* ...
    exp(-0.5 * sum(bsxfun(@times,inv_determ_r_r,...
                bsxfun(@times, opposite_r, ...
                    sqd_hs_sc_minus_mean_stack(1:num_samples, 1:num_samples, :)) ...
                + bsxfun(@times, opposite_r, ...
                    tr_sqd_hs_sc_minus_mean_stack(1:num_samples, 1:num_samples, :)) ...
                + prior_var_times_sqd_dist_stack_sc(1:num_samples, 1:num_samples, :)...
                ),3));
Ups_inv_K_r_r = solve_chol(R_r_s, Ups_r_r')';
inv_R_Ups_inv_K_r_r_s = R_r_s'\(Ups_inv_K_r_r * r_s);

% some code to test that this construction works          
% Lambda = diag(prior_sds.^2);
% W_qd = diag(qd_input_scales.^2);
% W_r = diag(r_input_scales.^2);
% mat = kron(ones(2),Lambda)+blkdiag(W_qd,W_r);
% 
% Ups_qd_r_test = @(i,j) qd_sqd_output_scale * r_sqd_output_scale *...
%     mvnpdf([hs_s(i,:)';hs_s(j,:)'],[prior.mean';prior.mean'],mat);
%
% As = [hs_sc_minus_mean_stack(3,:);hs_sc_minus_mean_stack(4,:)];
% Bs = 0*[prior.mean';prior.mean'];
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

% Generally, we don't have to worry about output scales that much, as they
% cancel. For example, we never need the output scale of delta.
% Unfortunately, we do need them when we are computing the variance. Below
% is a cheap hack to estimate the output scales over the transformed
% likelihoods. The plus eps bit is to manage the variance=0 resulting from
% a single sample.
tr_sqd_output_scale = r_sqd_output_scale * (var(tr_s)+eps)/(var(r_s)+eps);

lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

K_inv_K_r_sc_s = linsolve(R_r_s,linsolve(R_r_s, K_r_s_sc, lowr), uppr)';
mean_r_sc =  K_inv_K_r_sc_s * r_s;
mean_tr_sc = K_inv_K_r_sc_s * tr_s;

% use a crude thresholding here as our tilde transformation will fail if
% the mean goes below zero
mean_r_sc = max(mean_r_sc, eps);


% the difference between the mean of the transformed (log) likelihood and
% the transform of the mean likelihood
delta_tr_sc = mean_tr_sc - tilde(mean_r_sc, gamma_r);

del_inv_K = solve_chol(R_del_sc, delta_tr_sc)';
tr_inv_K = solve_chol(R_r_s, tr_s)';

% mean of int r(hs) p(hs) dhs given r_s
minty_r = ups_inv_K_r * r_s;

% mean of int delta(hs) r(hs) p(hs) dhs given r_s and delta_tr_sc
minty_del_r = del_inv_K * Ups_inv_K_del_r * r_s;

% mean of int tilde(r)(hs) r(hs) p(hs) dhs given r_s and tr_s 
minty_tr_r = tr_inv_K * Ups_inv_K_r_r * r_s;

% mean of int tilde(r)_0(hs) r(hs) p(hs) dhs given r_s and tr_s 
minty_tr0_r = minty_tr_r - minty_del_r;

% mean of int delta(hs) p(hs) dhs given r_s and delta_tr_sc 
minty_del = ups_inv_K_del * delta_tr_sc;

% the correction factor due to r being non-negative
correction = minty_del_r + gamma_r * minty_del;

% variance of int r(hs) p(hs) dhs given r_s
Vinty_r = chi_r - ups_inv_K_r * ups_r_s;
% NB: we actually want the variance of int tr(hs) p(hs) dhs, which differs
% only in the output scale used (the input scales are identical, and the
% actual observations tr_s and r_s do not enter). We will correct for using
% r_sqd_output_scale rather than tr_sqd_output_scale later.


% mean ev has been determined using the rescaled log-likelihoods (that have
% had the maximum log likelihood subtracted off), we return correct values
% by scaling back again)
mean_ev = minty_r + correction;
log_mean_evidence = max_log_r_s + log(mean_ev);

% (int dpsi_0/dtr(hs) m_{tr|s}(hs) dhs)^2
A = (minty_tr_r + gamma_r)^2;

% int int dpsi_0/dtr(hs) dpsi_0/dtr(hs') C_{tr|s}(hs, hs') dhs
% Note that this is the only term sensitive to the output scale for the
% transformed likelihood, tr: we simply rescale to account for it.
B = tr_sqd_output_scale/r_sqd_output_scale...
             * (gamma_r^2 * Vinty_r ...
             + 2 * gamma_r * (ups2_inv_K_r * r_s ...
                    - ups_inv_K_r * Ups_inv_K_r_r * r_s) ...
             + r_inv_K_chi3_inv_K_r ...
             - sum(inv_R_Ups_inv_K_r_r_s.^2)...
                );
            
% 2 (int dpsi_0/dtr(hs) m_{tr|s}(hs) dhs) * ...
%       (psi_0 - int dpsi_0/dtr(hs) tr_0(hs) dhs)
C = 2*(minty_tr_r + gamma_r)*(minty_r - minty_tr0_r - gamma_r);

% (psi_0 - int dpsi_0/dtr(hs) tr_0(hs) dhs)^2
D = (minty_r - minty_tr0_r - gamma_r)^2;

% int (psi_0 + int dpsi_0/dtr(hs) (tr(hs) - tr_0(hs)) dhs ...
%                   N(tr; m_{tr|s}, C_{tr|s}) dtr
mean_second_moment =  A + B + C + D;
log_mean_second_moment = 2*max_log_r_s + log(mean_second_moment);

var_ev = mean_second_moment - mean_ev.^2;
if var_ev < 0
    warning('variance of evidence negative');
    fprintf('variance of evidence: %g\n', var_ev.*exp(max_log_r_s)^2);
    var_ev = eps;
end
log_var_evidence = 2*max_log_r_s + log(var_ev);


% Store a lot of stuff in the r_gp_params structure.
% Ups_del_r has a different name in other files. 
Ups_sc_s = Ups_del_r;
names = {'candidate_locations', 'sqd_dist_stack_s', 'R_r_s', 'K_r_s', 'ups_r_s', ...
    'R_del_sc', 'K_del_sc', 'ups_del_sc', 'delta_tr_sc', 'jitters_r_s', ...
    'log_mean_second_moment', 'Ups_sc_s'};
for i = 1:length(names)
    r_gp_params.(names{i}) = eval(names{i});
end
