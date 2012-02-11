function vec = small_ups_vec(sqd_dist_stack_Amu, gp_A_hypers, prior)
% Returns the column vector
% % ups_s = int K(hs, hs_s) prior(hs) dhs
% = N(hs_s, mu, W_A + L);
% where hs_s is an element of A (forming the rows), which is modelled by a
% GP with sqd input scales W_A. 
% the prior is Gaussian with mean mu and variance L.

prior_var = diag(prior.covariance);

A_sqd_input_scales = exp(2*gp_A_hypers.log_input_scales);
A_sqd_output_scale = exp(2*gp_A_hypers.log_output_scale);

ups_log_input_scales = ...
    log(sqrt(prior_var + A_sqd_input_scales));

vec = gaussian_mat(sqd_dist_stack_Amu, ...
            A_sqd_output_scale, ups_log_input_scales);