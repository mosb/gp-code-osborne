function [xpc_unc] = expected_uncertainty_evidence...
                            (hs_a, sample_struct, prior_struct, r_gp, opt)
% returns the expected negative-squared-mean-evidence after adding a
% hyperparameter sample hs_a. This quantity is a scaled version of the
% expected variance in the evidence.
%
% [exp_log_unc] = ...
%   expected_uncertainty_evidence(hs_a, sample_struct, prior_struct, r_gp, opt)
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
% * NEEDS TO READ IN SO MUCH STUFF
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

num_sa = num_sa + 1;

opt.num_c = min(opt.num_c, num_s);
num_c = opt.num_c;

prior_var = prior_sds.^2;

[max_log_r_s, max_ind] = max(log_r_s);
% this function is only ever used to compare different hs_a's for the
% single fixed r_s, so no big deal about subtracting off this
log_r_s = log_r_s - max_log_r_s;
r_s = exp(log_r_s);


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
del_sqd_lambda = del_sqd_output_scale* ...
    prod(2*pi*del_input_scales.^2)^(-0.5);

sqd_r_input_scales = r_input_scales.^2;
sqd_del_input_scales = del_input_scales.^2;


hs_a_minus_mean = hs_a - prior_means;
sqd_hs_a_minus_mean = hs_a_minus_mean.^2;
tr_sqd_hs_sc_minus_mean_stack = tr(sqd_hs_sc_minus_mean_stack);


sum_prior_var_sqd_input_scales_r = ...
    prior_var + sqd_r_input_scales;
sum_prior_var_sqd_input_scales_del = ...
    prior_var + sqd_del_input_scales;

R_s = r_gp.R_s;

K_sa = nan(num_sa);
K_sa(num_sa, :) = K_sa_s;
K_sa(:, num_sa) = K_sa_s';
R_sa = updatechol(K_sa, R_s, num_sa);

yot_a = r_sqd_output_scale * ...
    prod(2*pi*sum_prior_var_sqd_input_scales_r)^(-0.5) * ...
    exp(-0.5 * ...
    sum(hs_a_minus_mean.^2./sum_prior_var_sqd_input_scales_stack_r));
yot_sa = [yot_s, yot_a];

yot_inv_K_r = solve_chol(R_sa, yot_sa)';


Yot_sc_a = 

Yot_sc_s = r_gp.Yot_sc_s;
Yot_sc_sa = [Yot_sc_s, Yot_sc_a];

del_inv_K_Yot_inv_K_sa = del_inv_K * solve_chol(R_r_sa, Yot_sc_sa')';


lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

sqd_dist_stack_s_a = bsxfun(@minus,...
                    reshape(hs_s,num_s,1,num_hps),...
                    reshape(hs_a,1,1,num_hps))...
                    .^2;  
K_r_s_a = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s_a, sqd_r_input_scales_stack), 3));

two_thirds_r = linsolve(R_r_s,linsolve(R_r_s, K_r_s_a, lowr), uppr)';
mean_r_sc =  two_thirds_r * r_s;
mean_tr_sc = two_thirds_r * tr_s;

n_a = 
tm_a = 
tv_a = 

n_r_s = n_s' * r_s;


xpc_unc = n_r_s^2 + ...
    2 * n_r_s * (gamma * exp(tm_a + 0.5*tv_a) - gamma) + ...
    n_a^2 * gamma^2 * ();
