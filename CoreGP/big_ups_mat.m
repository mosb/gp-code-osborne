function mat = big_ups_mat...
    (sqd_dist_stack_Amu, sqd_dist_stack_Bmu, sqd_dist_stack_AB, ...
    gp_A_hypers, gp_B_hypers, prior)
% Returns the matrix
% Ups_s_s' = int K(hs_s, hs) K(hs, hs_s') prior(hs) dhs
% = N([hs_s, hs_s'], [mu, mu], [W_A + L, L; L, W_B + L]);
% where hs_s is an element of A (forming the rows), which is modelled by a
% GP with sqd input scales W_A. 
% where hs_s' is an element of B (forming the cols), which is modelled by a
% GP with sqd input scales W_B.
% the prior is Gaussian with mean mu and variance L.

num_dims = size(sqd_dist_stack_Amu, 3);

sqd_input_scales_stack_A = ...
    reshape(exp(2*gp_A_hypers.log_input_scales), 1, 1, num_dims);
sqd_input_scales_stack_B = ...
    reshape(exp(2*gp_B_hypers.log_input_scales), 1, 1, num_dims);

sqd_output_scale_A = exp(2*gp_A_hypers.log_output_scales);
sqd_output_scale_B = exp(2*gp_B_hypers.log_output_scales);

prior_var_stack = reshape(diag(prior.covariance), 1, 1, num_dims);

prior_var_times_sqd_dist_stack_sca_a = bsxfun(@times, prior_var_stack, ...
                    sqd_dist_stack_AB);

opposite_A = sqd_input_scales_stack_A;
opposite_B = sqd_input_scales_stack_B;
inv_determ_del_r = (prior_var_stack.*(...
        sqd_input_scales_stack_B + sqd_input_scales_stack_A) + ...
        sqd_input_scales_stack_B.*sqd_input_scales_stack_A).^(-1);

% 2 pi is outside of sqrt because each element of determ is actually the
% determinant of a 2 x 2 matrix
mat = sqd_output_scale_A * sqd_output_scale_B * ...
    prod(1/(2*pi) * sqrt(inv_determ_del_r)) .* ...
    exp(-0.5 * sum(bsxfun(@times,inv_determ_del_r,...
                bsxfun(@times, opposite_B, ...
                    sqd_dist_stack_Amu) ...
                + bsxfun(@times, opposite_A, ...
                    sqd_dist_stack_Bmu) ...
                + prior_var_times_sqd_dist_stack_sca_a...
                ),3));

            
%     % some code to test that this construction works          
%     Lambda = diag(prior_var);
%     W_del = diag(del_input_scales.^2);
%     W_r = diag(logl_input_scales.^2);
%     mat = kron(ones(2),Lambda)+blkdiag(W_del,W_r);
% 
%     Ups_sca_a_test = @(i) del_sqd_output_scale * l_sqd_output_scale *...
%         mvnpdf([hs_sc(i,:)';new_sample_location'],[prior.mean';prior.mean'],mat);
