function [mean_out, sd_out, rhod, rhodd] = predict(X_star, gp, q_gp, r_gp, opt)
% function [mean, sd] = predict(X_star, gp, r_gp, opt)
% return the posterior mean and sd by marginalising hyperparameters.
% - X_star (n by d) is a matrix of the n (d-dimensional) points at which
% predictions are to be made
% - gp requires fields:
% * hyperparams(i).priorMean
% * hyperparams(i).priorSD
% * hypersamples.logL
% * hypersamples (if opt.prediction_model is gp or spgp)
% * hypersamples.hyperparameters (if using a handle for
% opt.prediction_model)
% - (optional) r_gp requires fields
% * quad_output_scale
% * quad_noise_sd
% * quad_input_scales
% alternatively: 
% [mean, sd] = predict(sample_struct, prior_struct, r_gp, opt)
% - sample_struct requires fields
% * samples
% * log_r
% and
% * mean_y
% * var_y
% or
% * qd
% * qdd
% or
% * q (if a posterior is required; returned in mean_out)
% - prior_struct requires fields
% * means
% * sds

if nargin<4
    opt = struct();
end

default_opt = struct('num_c', 100,...
                    'gamma_const', 100, ...
                    'num_box_scales', 3, ...
                    'prediction_model', 'spgp', ...
                    'no_adjustment', false, ...
                    'print', true);
                
names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end

% not fully optimised, further operations could be avoided if only the mean
% is required
want_sds = nargout > 1; 
 want_posterior = false;

if isstruct(X_star)
    sample_struct = X_star;
    prior_struct = gp;
    
    hs_s = sample_struct.samples;
    log_r_s = sample_struct.log_r;
    
    [num_s, num_hps] = size(hs_s);
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
        
    elseif isfield(sample_struct, 'q')
        % output argument will be a posterior, not a posterior mean
        want_sds = true;
        want_posterior = true;
        
        qd_s = sample_struct.q;
        qdd_s = sample_struct.q;
        
    end
        
    num_star = size(qd_s, 2);
    
    prior_means = prior_struct.means;
    prior_sds = prior_struct.sds;
    
    opt.prediction_model = 'arbitrary';
    
else
    [num_star] = size(X_star, 1);
    
    hs_s = vertcat(gp.hypersamples.hyperparameters);
    log_r_s = vertcat(gp.hypersamples.logL);
    
    [num_s, num_hps] = size(hs_s);
    
    prior_means = vertcat(gp.hyperparams.priorMean);
    prior_sds = vertcat(gp.hyperparams.priorSD);
    
    mean_y = nan(num_star, num_s);
    var_y = nan(num_star, num_s);

    if ischar(opt.prediction_model)
        switch opt.prediction_model
            case 'spgp'
                for hs = 1:num_s
                    [mean_y(:, hs), var_y(:, hs)] = ...
                        posterior_spgp(X_star,gp,hs,'var_not_cov');
                end
            case 'gp'
                for hs = 1:num_s
                    [mean_y(:, hs), var_y(:, hs)] = ...
                        posterior_gp(X_star,gp,hs,'var_not_cov');
                end
        end
    elseif isa(opt.prediction_model, 'function_handle')
        for hs = 1:num_s
            sample = gp.hypersamples(hs).hyperparameters;
            [mean_y(:, hs), var_y(:, hs)] = ...
                opt.prediction_model(X_star,sample);
        end
    end
    
    mean_y = mean_y';
    var_y = var_y';

    qd_s = mean_y;
    qdd_s = var_y + mean_y.^2;
    
end


opt.num_c = min(opt.num_c, num_s);
num_c = opt.num_c;

prior_sds_stack = reshape(prior_sds, 1, 1, num_hps);
prior_var_stack = prior_sds_stack.^2;

  
log_r_s = log_r_s - max(log_r_s);
r_s = exp(log_r_s);

mu_qd = mean(qd_s);
mu_qdd = mean(qdd_s);

qdmm_s = bsxfun(@minus, qd_s, mu_qd);
qddmm_s = bsxfun(@minus, qdd_s, mu_qdd);

if nargin<3 || isempty(q_gp)
    [q_noise_sd, q_input_scales, q_output_scale] = ...
        hp_heuristics(hs_s, qd_s, 10);

    sqd_output_scale = q_output_scale^2;
    q_input_scales = 10*q_input_scales;
else
    sqd_output_scale = q_gp.quad_output_scale^2;
    q_noise_sd =  q_gp.quad_noise_sd;
    q_input_scales = q_gp.quad_input_scales;
end

if nargin<4 || isempty(r_gp)
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
eps_sqd_lambda = sqd_output_scale* ...
    prod(2*pi*eps_input_scales.^2)^(-0.5);
r_noise_sd = r_noise_sd / sqrt(sqd_lambda);

lower_bound = min(hs_s) - opt.num_box_scales*input_scales;
upper_bound = max(hs_s) + opt.num_box_scales*input_scales;

% find the candidate points, far removed from existing samples
hs_c = find_farthest(hs_s, [lower_bound; upper_bound], num_c, ...
                            input_scales);
hs_sc = [hs_s; hs_c];
num_sc = size(hs_sc, 1);
num_c = num_sc - num_s;

sqd_dist_stack_sc = bsxfun(@minus,...
                    reshape(hs_sc,num_sc,1,num_hps),...
                    reshape(hs_sc,1,num_sc,num_hps))...
                    .^2;  
sqd_dist_stack_s = sqd_dist_stack_sc(1:num_s, 1:num_s, :);

sqd_input_scales_stack = reshape(input_scales.^2,1,1,num_hps);
sqd_eps_input_scales_stack = reshape(eps_input_scales.^2,1,1,num_hps);
                
K_s = sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s, sqd_input_scales_stack), 3)); 
K_rs = improve_covariance_conditioning(K_s, r_s, 10^-16);
R_rs = chol(K_rs);
K_qds = improve_covariance_conditioning(K_s, qdmm_s, 10^-16);
R_qds = chol(K_qds);
K_qdds = improve_covariance_conditioning(K_s, qddmm_s, 10^-16);
R_qdds = chol(K_qdds);

K_eps = eps_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_sc, sqd_eps_input_scales_stack), 3)); 
importance_sc = ones(num_sc,1);
importance_sc(num_s + 1 : end) = 10;
K_eps = improve_covariance_conditioning(K_eps, importance_sc, 10^-16);
R_eps = chol(K_eps);     

sqd_dist_stack_s_sc = sqd_dist_stack_sc(1:num_s, :, :);
K_s_sc = sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s_sc, sqd_input_scales_stack), 3));  
             
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

yot_inv_K_rs = solve_chol(R_rs, yot_s)';
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

inv_K_Yot_inv_K_qds = solve_chol(R_qds, solve_chol(R_rs, Yot_s)')';
inv_K_Yot_inv_K_qdds = solve_chol(R_qdds, solve_chol(R_rs, Yot_s)')';
inv_K_Yot_inv_K_qds_eps = solve_chol(R_eps, solve_chol(R_qds, Yot_s_eps)')';
inv_K_Yot_inv_K_qdds_eps = solve_chol(R_eps, solve_chol(R_qdds, Yot_s_eps)')';
          

tilde = @(x, gamma_x) log(bsxfun(@rdivide, x, gamma_x) + 1);
%inv_tilda = @(tx, gamma_x) exp(bsxfun(@plus, tx, log(gamma_x))) - gamma_x;

gamma_r = opt.gamma_const;
tr_s = tilde(r_s, gamma_r);

gamma_qdd = opt.gamma_const*max(eps,max(qdd_s));
tqdd_s = tilde(qdd_s, gamma_qdd);


lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

two_thirds_r = linsolve(R_rs,linsolve(R_rs, K_s_sc, lowr), uppr)';
two_thirds_qd = linsolve(R_qds,linsolve(R_qds, K_s_sc, lowr), uppr)';
two_thirds_qdd = linsolve(R_qdds,linsolve(R_qdds, K_s_sc, lowr), uppr)';

mean_r_sc =  two_thirds_r * r_s;
mean_qd_sc = bsxfun(@plus, mu_qd, two_thirds_qd * qdmm_s);
mean_qdd_sc = bsxfun(@plus, mu_qdd, two_thirds_qdd * qddmm_s);

mean_tr_sc = two_thirds_r * tr_s;
mean_tqdd_sc = two_thirds_qdd * tqdd_s;

%c_inds = num_sc - (num_c-1:-1:0);

% Delta_tr = zeros(num_sc, 1);
% eps_rr_sc = zeros(num_sc, 1);
% Delta_tqdd = zeros(num_sc, num_star);
% eps_qdr_sc = zeros(num_sc, num_star);
% eps_rqdd_sc = zeros(num_sc, num_star);
% eps_qddr_sc = zeros(num_sc, num_star);

% use a crude thresholding here as our tilde transformation will fail if
% the mean 
mean_r_sc = max(eps, mean_r_sc);
mean_qdd_sc = max(eps, mean_qdd_sc);

Delta_tr = mean_tr_sc - tilde(mean_r_sc, gamma_r);
Delta_tqdd = mean_tqdd_sc - tilde(mean_qdd_sc, gamma_qdd);

eps_rr_sc = mean_r_sc .* Delta_tr;
eps_qdr_sc = bsxfun(@times, mean_qd_sc, Delta_tr);
eps_rqdd_sc = bsxfun(@times, mean_r_sc, Delta_tqdd);
eps_qddr_sc = bsxfun(@times, mean_qdd_sc, Delta_tr);

minty_r = yot_inv_K_rs * r_s;
minty_Delta_tr = yot_inv_K_eps * Delta_tr;
minty_eps_rr = yot_inv_K_eps * eps_rr_sc;
minty_eps_qdr = (yot_inv_K_eps * eps_qdr_sc)';
minty_eps_rqdd = (yot_inv_K_eps * eps_rqdd_sc)';
minty_eps_qddr = (yot_inv_K_eps * eps_qddr_sc)';

% all the quantities below need to be adjusted to account for the non-zero
% prior means of qd and qdd
minty_qd_r = qdmm_s' * inv_K_Yot_inv_K_qds * r_s + ...
                mu_qd' * minty_r;
rhod = minty_qd_r / minty_r;
minty_qd_eps_rr = qdmm_s' * inv_K_Yot_inv_K_qds_eps * eps_rr_sc + ...
                mu_qd' * minty_eps_rr;

if want_sds               
minty_qdd_r = qddmm_s' * inv_K_Yot_inv_K_qdds * r_s + ...
                mu_qdd' * minty_r;
rhodd = minty_qdd_r / minty_r;
% only need the diagonals of this quantity, the full covariance is not
% required
minty_qdd_eps_rqdd = ...
    sum((qddmm_s' * inv_K_Yot_inv_K_qdds_eps) .* eps_rqdd_sc', 2) + ...
                mu_qdd' .* minty_eps_rqdd;
minty_qdd_eps_rr = qddmm_s' * inv_K_Yot_inv_K_qdds_eps * eps_rr_sc + ...
                mu_qdd' * minty_eps_rr;
end




adj_rhod_tr = (minty_qd_eps_rr + gamma_r * minty_eps_qdr ...
                -(minty_eps_rr + gamma_r * minty_Delta_tr) * rhod) / minty_r;
if want_sds            
adj_rhodd_tq = (minty_qdd_eps_rqdd + gamma_qdd * minty_eps_rqdd) / minty_r;
adj_rhodd_tr = (minty_qdd_eps_rr + gamma_r * minty_eps_qddr ...
    -(minty_eps_rr + gamma_r * minty_Delta_tr) * rhodd) / minty_r;
end
if opt.no_adjustment
    adj_rhod_tr = 0;
    adj_rhodd_tq = 0;
    adj_rhodd_tr = 0;
end


mean_out = rhod + adj_rhod_tr;
if want_sds
second_moment = rhodd + adj_rhodd_tq + adj_rhodd_tr;
if want_posterior
    mean_out = second_moment;
    sd_out = nan;
else
    sd_out = sqrt(second_moment - mean_out.^2);
end
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