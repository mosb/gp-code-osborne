function [xpc_unc, tm_a, tv_a] = expected_uncertainty_evidence...
      (new_sample_location, samples, prior, r_gp_params, opt)
% returns the expected negative-squared-mean-evidence after adding a
% hyperparameter sample hs_a. This quantity is a scaled version of the
% expected variance in the evidence.
%
% [exp_log_unc] = ...
%   expected_uncertainty_evidence(hs_a, sample_struct, prior_struct, r_gp_params, widths_quad_input_scales, opt)
% - hs_a (1 by the number of hyperparameters) is a row vector expressing
%   the relevant trial hyperparameters. If hs_a is empty or omitted, then the
%   returned mean and variance are the non-expected qantities for the
%   evidence.
% - evidence: the current evidence
% - sample_struct requires fields
%   * samples
%   * log_r
% - prior_struct requires fields
%   * means
%   * sds
% - input r_gp_params has fields
%   * quad_output_scale
%   * quad_noise_sd
%   * quad_input_scales
%   * candidate_locations
%   * sqd_dist_stack_s
%   * R_r_s
%   * K_r_s
%   * ups_r_s
%   * R_del_sc
%   * K_del_sc
%   * ups_del_sc
%   * delta_tr_sc
%   * jitters_r_s
%   * log_mean_second_moment
%   * Ups_sc_s
%   * minty_del
%   * tr_sqd_output_scale
      
new_sample_location = new_sample_location(:)';  % Hack for fmincon.

if nargin<5
    opt = struct();
end

opt = struct('gamma_const', (exp(1)-1)^(-1), ... % numerical scaling factor
                    'allowed_cond_error',10^-14, ... % allowed conditioning error
                    'sds_tr_input_scales', opt.sds_tr_input_scales,...
                    'delta_update', false);
% sds_tr_input_scales represents the posterior standard deviations in the
% input scales for tr. If false, a delta function posterior is assumed.            
%opt = set_defaults( opt, default_opt );

log_r_s = samples.log_r;

[num_s, num_hps] = size(samples.locations);

% David asking Mike:  Should this be here?
% Mike says: yes, the expecetd variance depends on the squared mean for the
% integral over the likelihood, for which we need hs_c.
hs_sc = [samples.locations; r_gp_params.candidate_locations];
hs_sca = [hs_sc; new_sample_location];
hs_sa = [samples.locations; new_sample_location];

num_sc = size(hs_sc, 1);
num_sca = num_sc + 1;
num_sa = num_s + 1;
range_sa = [1:num_s,num_sca];

prior_var = diag(prior.covariance)';
prior_var_stack = reshape(prior_var, 1, 1, num_hps);

[max_log_r_s, max_ind] = max(log_r_s);
% this function is only ever used to compare different hs_a's for the
% single fixed r_s, so no big deal about subtracting off this
log_r_s = log_r_s - max_log_r_s;
r_s = exp(log_r_s);


tilde = @(x, gamma_r_x) log(bsxfun(@rdivide, x, gamma_r_x) + 1);
%inv_tilda = @(tx, gamma_r_x) exp(bsxfun(@plus, tx, log(gamma_r_x))) - gamma_r_x;

% this is after r_s has already been divided by exp(max_log_r_s). tr_s is
% its correct value, but log(gamma_r) has effectively had max_log_r_s
% subtracted from it
gamma_r = opt.gamma_const; 
tr_s = tilde(r_s, gamma_r);


% hyperparameters for gp over the likelihood, r, assumed to have zero
% mean
r_sqd_output_scale = r_gp_params.quad_output_scale^2;
r_input_scales = r_gp_params.quad_input_scales;
r_sqd_lambda = r_sqd_output_scale* ...
    prod(2*pi*r_input_scales.^2)^(-0.5);

sqd_r_input_scales = r_input_scales.^2;
sqd_r_input_scales_stack = reshape(sqd_r_input_scales,1,1,num_hps);

% hyperparameters for gp over delta, the difference between log-gp-mean-r and
% gp-mean-log-r
del_input_scales = 0.5 * r_input_scales;
del_sqd_output_scale = 0.1 * r_sqd_output_scale;
del_sqd_lambda = del_sqd_output_scale* ...
    prod(2*pi*del_input_scales.^2)^(-0.5);

sqd_del_input_scales = del_input_scales.^2;
sqd_del_input_scales_stack = reshape(del_input_scales.^2,1,1,num_hps);

hs_a_minus_mean = new_sample_location - prior.mean;
hs_sca_minus_mean_stack = reshape(bsxfun(@minus, hs_sca, prior.mean),...
                    num_sca, 1, num_hps);
sqd_hs_sca_minus_mean_stack = ...
    repmat(hs_sca_minus_mean_stack.^2, [1, 1, 1]);
hs_a_minus_mean_stack = reshape(hs_a_minus_mean,...
                    1, 1, num_hps);
sqd_hs_a_minus_mean_stack = ...
    repmat(hs_a_minus_mean_stack.^2, [num_sca, 1, 1]);

sqd_dist_stack_sca_a = reshape(bsxfun(@minus, hs_sca, new_sample_location).^2, ...
                    num_sca, 1, num_hps);
sqd_dist_stack_sa_a = reshape(bsxfun(@minus, hs_sa, new_sample_location).^2, ...
                    num_sa, 1, num_hps);

% we update the covariance matrix over r. We consider all old jitters fixed
% and add in jitter at hs_a sufficient to render the matrix
% well-conditioned (maybe we should revisit old jitters, even if it'd be
% slower?).
K_r_sa = nan(num_sa);
diag_K_r_sa = diag_inds(K_r_sa);
K_r_sa(diag_K_r_sa(1:num_s)) = diag(r_gp_params.K_r_s);
K_r_sa_a = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_sa_a, sqd_r_input_scales_stack), 3));
K_r_sa(num_sa, :) = K_r_sa_a;
K_r_sa(:, num_sa) = K_r_sa_a';

% this importances vector is to force the jitter to be applied solely to
% the added point hs_a. improve_covariance_conditioning will automatically
% do this so long as K_r_sa has nans in the appropriate off-diagonal
% elements, but not if K_r_sa is 2x2, so that there are no off-diagonal
% elements.
importances = [inf(num_s,1);0];
K_r_sa = improve_covariance_conditioning(K_r_sa, ...
    importances, opt.allowed_cond_error);
R_r_sa = updatechol(K_r_sa, r_gp_params.R_r_s, num_sa);

% ups_s = int K(hs, samples.locations)  prior(hs) dhs

sum_prior_var_sqd_input_scales_r = ...
    prior_var + sqd_r_input_scales;
ups_r_a = r_sqd_output_scale * ...
    prod(2*pi*sum_prior_var_sqd_input_scales_r)^(-0.5) * ...
    exp(-0.5 * ...
    sum(hs_a_minus_mean.^2./sum_prior_var_sqd_input_scales_r));

ups_r_sa = [r_gp_params.ups_r_s; ups_r_a];
ups_inv_K_r_sa = solve_chol(R_r_sa, ups_r_sa)';

if opt.delta_update
    K_del_sca = nan(num_sca);
    diag_K_del_sca = diag_inds(K_del_sca);
    K_del_sca(diag_K_del_sca(1:num_sc)) = diag(r_gp_params.K_del_sc);
    K_del_sca_a = del_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                        sqd_dist_stack_sca_a, sqd_del_input_scales_stack), 3));
    K_del_sca(num_sca, :) = K_del_sca_a;
    K_del_sca(:, num_sca) = K_del_sca_a';


    importances = [inf(num_sc,1);0];
    K_del_sca = improve_covariance_conditioning(K_del_sca, ...
        importances, opt.allowed_cond_error);
    R_del_sca = updatechol(K_del_sca, r_gp_params.R_del_sc, num_sca); 

    % we add noise to delta to account for the fact that it will change
    % following the addition of a new observation. delta will be unchanged at
    % zero at hs_s, and will change at hs_c more for locations close to hs_a.
    % Of course, we also know with certainty that delta at hs_a will be zero. 
    del_noise = K_del_sca_a;
    del_noise(1:num_s) = 0;
    del_noise(num_sca) = 0;
    R_del_sca = perturbchol(R_del_sca, del_noise);
    
    sum_prior_var_sqd_input_scales_del = ...
    prior_var + sqd_del_input_scales;
    ups_del_a = del_sqd_output_scale * ...
        prod(2*pi*sum_prior_var_sqd_input_scales_del)^(-0.5) * ...
        exp(-0.5 * ...
        sum(hs_a_minus_mean.^2./sum_prior_var_sqd_input_scales_del));
    
    ups_del_sca = [r_gp_params.ups_del_sc; ups_del_a];
    ups_inv_K_del_sca = solve_chol(R_del_sca, ups_del_sca)';  
end

                
prior_var_times_sqd_dist_stack_sca_a = bsxfun(@times, prior_var_stack, ...
                    sqd_dist_stack_sca_a);
                
opposite_del = sqd_del_input_scales_stack;
opposite_r = sqd_r_input_scales_stack;
sqd_r_input_scales_stack = reshape(r_input_scales.^2,1,1,num_hps);
sqd_del_input_scales_stack = reshape(del_input_scales.^2,1,1,num_hps);
inv_determ_del_r = (prior_var_stack.*(...
        sqd_r_input_scales_stack + sqd_del_input_scales_stack) + ...
        sqd_r_input_scales_stack.*sqd_del_input_scales_stack).^(-1);

% Ups_s_s = int K(samples.locations, hs) K(hs, samples.locations) prior(hs) dhs
Ups_sca_a = del_sqd_output_scale * r_sqd_output_scale * ...
    prod(1/(2*pi) * sqrt(inv_determ_del_r)) .* ...
    exp(-0.5 * sum(bsxfun(@times,inv_determ_del_r,...
                bsxfun(@times, opposite_r, ...
                    sqd_hs_sca_minus_mean_stack) ...
                + bsxfun(@times, opposite_del, ...
                    sqd_hs_a_minus_mean_stack) ...
                + prior_var_times_sqd_dist_stack_sca_a...
                ),3));
% 2 pi is outside of sqrt because each element of determ is actually the
% determinant of a 2 x 2 matrix
            
%     % some code to test that this construction works          
%     Lambda = diag(prior_var);
%     W_del = diag(del_input_scales.^2);
%     W_r = diag(r_input_scales.^2);
%     mat = kron(ones(2),Lambda)+blkdiag(W_del,W_r);
% 
%     Ups_sca_a_test = @(i) del_sqd_output_scale * r_sqd_output_scale *...
%         mvnpdf([hs_sc(i,:)';new_sample_location'],[prior.mean';prior.mean'],mat);



if opt.delta_update
    % update for the influence of the new observation at hs_a on delta.
    Ups_sca_sa = [r_gp_params.Ups_sc_s, Ups_sca_a(1:num_sc,:);
                    Ups_sca_a(range_sa,:)'];
    delta_tr_sca = [r_gp_params.delta_tr_sc;0];
    del_inv_K = solve_chol(R_del_sca, delta_tr_sca)';
    del_inv_K_Ups_inv_K_r_sa = del_inv_K * solve_chol(R_r_sa, Ups_sca_sa')';

    minty_del = ups_inv_K_del_sca * delta_tr_sca;
else     
    Ups_sc_sa = [r_gp_params.Ups_sc_s,Ups_sca_a(1:num_sc,:)];
    del_inv_K_Ups_inv_K_r_sa = ...
        r_gp_params.del_inv_K * solve_chol(R_r_sa, Ups_sc_sa')';
    minty_del = r_gp_params.minty_del;
end

n_sa = del_inv_K_Ups_inv_K_r_sa + ups_inv_K_r_sa;

lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

sqd_dist_s_a = bsxfun(@minus, samples.locations, new_sample_location).^2;  
K_r_s_a = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_s_a, sqd_r_input_scales), 2));
               
%remove the jitter associated with the closest datum to to hs_a
% [K_r_s, R_r_s] = ...
%     jitter_correction(r_gp_params.jitters_r_s, K_r_s_a', r_gp_params.K_r_s, r_gp_params.R_r_s);
K_r_s = r_gp_params.K_r_s;
R_r_s = r_gp_params.R_r_s;


invR_K_r_s_a = linsolve(R_r_s, K_r_s_a, lowr);                
K_invK_a_s = linsolve(R_r_s, invR_K_r_s_a, uppr)';



n_a = n_sa(num_sa);
% zero prior mean
tm_a = K_invK_a_s * tr_s;

tv_a = r_sqd_lambda - invR_K_r_s_a' * invR_K_r_s_a;
if tv_a < 0
    tv_a = eps;
end

% Generally, we don't have to worry about output scales that much, as they
% cancel.
% Unfortunately, we do need them when we are computing the variance. Below
% is a cheap hack to estimate the output scales over the transformed
% likelihoods. The plus eps bit is to manage the variance=0 resulting from
% a single sample.
tr_sqd_output_scale = r_gp_params.tr_sqd_output_scale;
tv_a = tr_sqd_output_scale/r_sqd_output_scale * tv_a;

if opt.sds_tr_input_scales
    % we correct for the impact of learning this new hyperparameter sample,
    % r_a, on our belief about the input scales
    
    % the variances of our posteriors over our input scales. We assume the
    % covariance matrix has zero off-diagonal elements; the posterior is
    % spherical. 
    V_theta = opt.sds_tr_input_scales.^2;
    if size(V_theta,1) == 1
        V_theta = V_theta';
    end
    
    
    invK_tr_s = solve_chol(R_r_s, tr_s);

    sqd_dist_stack_s_a = reshape(sqd_dist_s_a', 1, num_s, num_hps);

    % Dtheta_ prefix denotes the derivative of a quantity: each plate is the
    % derivative with respect to a different log input scale
    
    Dtheta_K_a_s = bsxfun(@times, K_r_s_a', ...
        bsxfun(@rdivide, ...
        sqd_dist_stack_s_a, ...
        sqd_r_input_scales_stack));
    Dtheta_K_r_s = bsxfun(@times, K_r_s, ...
        bsxfun(@rdivide, ...
        r_gp_params.sqd_dist_stack_s, ...
        sqd_r_input_scales_stack));

    
    Dtheta_tm_a = prod3(Dtheta_K_a_s, invK_tr_s) ...
            - prod3(K_invK_a_s, prod3(Dtheta_K_r_s, invK_tr_s));
        
    tv_a = tv_a + sum(reshape(Dtheta_tm_a.^2, num_hps, 1 , 1) .* V_theta);
end



n_r_s = n_sa(1:num_s) * r_s + gamma_r * minty_del;

xpc_unc =  exp(r_gp_params.log_mean_second_moment)...
    - exp(2*max_log_r_s) * (n_r_s^2 ...
    + 2 * n_r_s * n_a * (gamma_r * exp(tm_a + 0.5*tv_a) - gamma_r) ...
    + n_a^2 * gamma_r^2 * ...
        (exp(2*tm_a + 2*tv_a) - 2 * exp(tm_a + 0.5*tv_a) + 1));
    
