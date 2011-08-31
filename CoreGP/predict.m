function [mean_out, sd_out] = predict(X_star, gp, r_gp, opt)
% function [mean, sd] = predict(X_star, gp, r_gp, opt)
% return the posterior mean and sd by marginalising hyperparameters

[num_star, num_dims] = size(X_star);

hs_s = vertcat(gp.hypersamples.hyperparameters);
[num_s, num_hps] = size(hs_s);

if nargin<4
    opt = struct();
end
% not fully optimised, further operations could be avoided if only the mean
% is required
want_sds = nargout > 1; 

default_opt = struct('num_c', min(num_s, 500),...
                    'gamma_const', 100, ...
                    'num_box_scales', 3, ...
                    'sparse', true, ...
                    'print', true);
                
names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end

num_c = opt.num_c;


prior_means = vertcat(gp.hyperparams.priorMean);
prior_sds = vertcat(gp.hyperparams.priorSD);

prior_sds_stack = reshape(prior_sds, 1, 1, num_hps);
prior_var_stack = prior_sds_stack.^2;



              
log_r_s = vertcat(gp.hypersamples.logL);
log_r_s = log_r_s - max(log_r_s);
r_s = exp(log_r_s);



if nargin<3
    [r_noise_sd, r_input_scales, r_output_scale] = ...
        hp_heuristics(hs_s, r_s, 10);

    sqd_output_scale = r_output_scale^2;
    r_input_scales = 10*r_input_scales;
else
    sqd_output_scale = r_gp.quad_output_scale^2;
    r_noise_sd =  r_gp.quad_noise_sd;
    r_input_scales = r_gp.quad_input_scales;
end

% we force GPs for r, qd, qdd, tr, and tqdd to share the same input scales.
% eps_rr, eps_qdr, eps_rqdd, eps_qddr are assumed to all have input scales
% equal to half of those for r.

input_scales = r_input_scales;
eps_input_scales = 0.5 * input_scales;

sqd_lambda = sqd_output_scale* ...
    prod(2*pi*input_scales.^2)^(-0.5);
r_noise_sd = r_noise_sd / sqrt(sqd_lambda);

lower_bound = min(hs_s) - opt.num_box_scales*input_scales;
upper_bound = max(hs_s) + opt.num_box_scales*input_scales;

% find the candidate points, far removed frome xisting samples
hs_c = far_pts(hs_s, [lower_bound; upper_bound], num_c);
hs_sc = [hs_s; hs_c];
num_sc = size(hs_sc, 1);

sqd_dist_stack_sc = bsxfun(@minus,...
                    reshape(hs_sc,num_sc,1,num_hps),...
                    reshape(hs_sc,1,num_sc,num_hps))...
                    .^2;  
sqd_dist_stack_s = sqd_dist_stack_sc(1:num_s, 1:num_s, :);

sqd_input_scales_stack = reshape(input_scales.^2,1,1,num_hps);
sqd_eps_input_scales_stack = reshape(eps_input_scales.^2,1,1,num_hps);
                
K_s = sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s, sqd_input_scales_stack), 3)); 
K_s = improve_covariance_conditioning(K_s);
R_s = chol(K_s);

K_eps = sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_sc, sqd_eps_input_scales_stack), 3)); 
K_eps = improve_covariance_conditioning(K_eps);
R_eps = chol(K_eps);     

sqd_dist_stack_s_c = sqd_dist_stack_sc(1:num_s, num_s+1:end, :);
K_s_c = sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s_c, sqd_input_scales_stack), 3));  
             
sum_prior_var_sqd_input_scales_stack = ...
    prior_var_stack + sqd_input_scales_stack;
determ = 2*prior_var_stack.*sqd_input_scales_stack + ...
        sqd_input_scales_stack.^2;
PrecX = sum_prior_var_sqd_input_scales_stack./determ;
PrecY = -prior_var_stack./determ;
const = 1/(2*pi*sqrt(determ));

sum_prior_var_sqd_input_scales_stack_eps = ...
    prior_var_stack + sqd_eps_input_scales_stack;
determ_s_eps = prior_var_stack.*(...
        sqd_eps_input_scales_stack + sqd_input_scales_stack) + ...
        sqd_eps_input_scales_stack.*sqd_input_scales_stack;
% NB: the inversion switches the eps & s for the following two terms
PrecX_s = sqd_eps_input_scales_stack./determ_s_eps;
PrecX_eps = sqd_input_scales_stack./determ_s_eps;
PrecY_s_eps = prior_var_stack./determ_s_eps;
% 2 pi is outside here because each element of determ_s_eps is actually the
% determinant of a 2 x 2 matrix
const_s_eps  = 1/(2*pi*sqrt(determ_s_eps));
    
hs_sc_minus_mean_stack = reshape(bsxfun(@minus, hs_sc, prior_means'),...
                    num_sc, 1, num_hps);
sqd_hs_sc_minus_mean_stack = ...
    repmat(hs_sc_minus_mean_stack.^2, 1, num_sc, 1);
tr_sqd_hs_sc_minus_mean_stack = tr(sqd_hs_sc_minus_mean_stack);
sum_sqd_hs_sc_minus_mean_stack = sqd_hs_sc_minus_mean_stack + ...
    tr_sqd_hs_sc_minus_mean_stack;
prod_hs_sc_minus_mean_stack = ...
    bsxfun(@(x,y) x.*y, ...
    hs_sc_minus_mean_stack, tr(hs_sc_minus_mean_stack));

yot_s = sqd_output_scale * ...
    prod(2*pi*sum_prior_var_sqd_input_scales_stack)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, hs_sc_minus_mean_stack(1:num_s, :, :).^2, ...
    sum_prior_var_sqd_input_scales_stack),3));
yot_eps = sqd_output_scale * ...
    prod(2*pi*sum_prior_var_sqd_input_scales_stack_eps)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, hs_sc_minus_mean_stack.^2, ...
    sum_prior_var_sqd_input_scales_stack_eps),3));

yot_inv_K_s = solve_chol(R_s, yot_s)';
yot_inv_K_eps = solve_chol(R_eps, yot_eps)';
                
Yot_s = sqd_output_scale.^2 * prod(bsxfun(@times, const, ...
    exp(-0.5 * bsxfun(@times, PrecX, ...
                    sum_sqd_hs_sc_minus_mean_stack(1:num_s, 1:num_s, :)) ...
              -bsxfun(@times, PrecY, ...
                    prod_hs_sc_minus_mean_stack(1:num_s, 1:num_s, :)))),3);   
Yot_s_eps = sqd_output_scale.^2 * prod(const_s_eps) * ...
    exp(-0.5 * sum(bsxfun(@times, PrecX_s, ...
                    sqd_hs_sc_minus_mean_stack(1:num_s, :, :)) ...
                + bsxfun(@times, PrecX_eps, ...
                    tr_sqd_hs_sc_minus_mean_stack(1:num_s, :, :)) ...
                + bsxfun(@times, PrecY_s_eps, ...
                    sqd_dist_stack_sc(1:num_s,:,:))...
                ,3));
          

% As = [hs_sc_minus_mean_stack(3,:);hs_sc_minus_mean_stack(4,:)];
% Bs = 0*[prior_means';prior_means'];
% scalesss = [input_scales;eps_input_scales].^2;
% covmat = kron2d(diag(prior_sds.^2), ones(2)) + diag(scalesss(:));
% sqd_output_scale.^2 * mvnpdf(As(:),Bs(:),covmat);

inv_K_Yot_inv_K_s = solve_chol(R_s, solve_chol(R_s, Yot_s)')';
inv_K_Yot_inv_K_s_eps = solve_chol(R_eps, solve_chol(R_s, Yot_s_eps)')';
           
mean_y = nan(num_star, num_s);
var_y = nan(num_star, num_s);
for hs = 1:num_s
    if opt.sparse
        [mean_y(:, hs), var_y(:, hs)] = ...
            posterior_spgp(X_star,gp,hs,'var_not_cov');
    else
        [mean_y(:, hs), var_y(:, hs)] = ...
            posterior_gp(X_star,gp,hs,'var_not_cov');
    end

end

mean_y = mean_y';
var_y = var_y';

qd_s = mean_y;
qdd_s = var_y + mean_y.^2;

mu_qd = mean(qd_s);
mu_qdd = mean(qdd_s);

qd_s = bsxfun(@minus, qd_s, mu_qd);
qdd_s = bsxfun(@minus, qdd_s, mu_qdd);

tilde = @(x, gamma_x) log(bsxfun(@rdivide, x, gamma_x) + 1);
%inv_tilda = @(tx, gamma_x) exp(bsxfun(@plus, tx, log(gamma_x))) - gamma_x;

gamma_r = opt.gamma_const;
tr_s = tilde(r_s, gamma_r);

gamma_qdd = opt.gamma_const*max(qdd_s);
tqdd_s = tilde(qdd_s, gamma_qdd);



lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

two_thirds = linsolve(R_s,linsolve(R_s, K_s_c, lowr), uppr)';

mean_r_c =  two_thirds * r_s;
mean_qd_c = bsxfun(@plus, mu_qd, two_thirds * qd_s);
mean_qdd_c = bsxfun(@plus, mu_qdd, two_thirds * qdd_s);

mean_tr_c = two_thirds * tr_s;
mean_tqdd_c = two_thirds * tqdd_s;

c_inds = num_sc - (num_c-1:-1:0);

Delta_tr = zeros(num_sc, 1);
eps_rr_sc = zeros(num_sc, 1);
Delta_qdd = zeros(num_sc, num_star);
eps_qdr_sc = zeros(num_sc, num_star);
eps_rqdd_sc = zeros(num_sc, num_star);
eps_qddr_sc = zeros(num_sc, num_star);

Delta_tr(c_inds, :) = mean_tr_c - tilde(mean_r_c, gamma_r);
Delta_qdd(c_inds, :) = mean_tqdd_c - tilde(mean_qdd_c, gamma_qdd);

eps_rr_sc(c_inds) = mean_r_c .* Delta_tr(c_inds);
eps_qdr_sc(c_inds, :) = bsxfun(@times, mean_qd_c, Delta_tr(c_inds));
eps_rqdd_sc(c_inds, :) = bsxfun(@times, mean_r_c, Delta_qdd(c_inds, :));
eps_qddr_sc(c_inds, :) = bsxfun(@times, mean_qdd_c, Delta_tr(c_inds));

minty_r = yot_inv_K_s * r_s;
minty_Delta_tr = yot_inv_K_eps * Delta_tr;
minty_eps_rr = yot_inv_K_eps * eps_rr_sc;
minty_eps_qdr = (yot_inv_K_eps * eps_qdr_sc)';
minty_eps_rqdd = (yot_inv_K_eps * eps_rqdd_sc)';
minty_eps_qddr = (yot_inv_K_eps * eps_qddr_sc)';

% all the quantities below need to be adjusted to account for the non-zero
% prior means of qd and qdd
minty_qd_r = qd_s' * inv_K_Yot_inv_K_s * r_s + ...
                mu_qd' * minty_r;
rhod = minty_qd_r / minty_r;
minty_qd_eps_rr = qd_s' * inv_K_Yot_inv_K_s_eps * eps_rr_sc + ...
                mu_qd' * minty_eps_rr;

if want_sds               
minty_qdd_r = qdd_s' * inv_K_Yot_inv_K_s * r_s + ...
                mu_qdd' * minty_r;
rhodd = minty_qdd_r / minty_r;
% only need the diagonals of this quantity, the full covariance is not
% required
minty_qdd_eps_rqdd = ...
    sum((qdd_s' * inv_K_Yot_inv_K_s_eps) .* eps_rqdd_sc', 2) + ...
                mu_qdd' .* minty_eps_rqdd;
minty_qdd_eps_rr = qdd_s' * inv_K_Yot_inv_K_s_eps * eps_rr_sc + ...
                mu_qdd' * minty_eps_rr;
end




adj_rhod_tr = (minty_qd_eps_rr + gamma_r * minty_eps_qdr ...
                -(minty_eps_rr + gamma_r * minty_Delta_tr) * rhod) / minty_r;
if want_sds            
adj_rhodd_tq = (minty_qdd_eps_rqdd + gamma_qdd * minty_eps_rqdd) / minty_r;
adj_rhodd_tr = (minty_qdd_eps_rr + gamma_r * minty_eps_qddr ...
    -(minty_eps_rr + gamma_r * minty_Delta_tr) * rhodd) / minty_r;
end


mean_out = rhod + adj_rhod_tr;
if want_sds
second_moment = rhodd + adj_rhodd_tq + adj_rhodd_tr;
sd_out = sqrt(second_moment - mean_out.^2);
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