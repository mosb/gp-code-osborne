function [x_zoomed, local_optimum_flag, y_zoomed] = simple_zoom_pt(x, grad, input_scales, min_max_local_optimum_flag)
% find the approximate location of the local minimum near to x. We assume a
% constant mean function, Gaussian covariance and noiseless observations.
% x: point to `zoom' from
% grad: gradient at x
% input_scale: input scale for gp over f(x)
% best_ind: hyperparameter sample index to use
% x_zoomed: location of local minimum `zoomed' to
% y_zoomed: value of local minimum `zoomed' to (assuming the value at x was
% zero)
% local_optimum_flag: point unmoved, likely a local minimum
% 
% NB: due to the assumed covariance, a product of independent terms over
% each input dimension, the zoomed_x will not, in general, be shifted by a
% multiple of the grad.*input_scales from x; the input scales being
% identical giving an exception.

inv_sqd_input_scales = input_scales'.^-2;

num_pts = size(x,1);
num_dims = size(x,2);

if size(grad,2) ~= num_dims
    grad = grad';
end

if nargin<4
    min_max_local_optimum_flag = 'minimise';
end

x(x == -inf) = -2^1023;
x(x == inf) = 2^1023;

Kmat = diag([1;inv_sqd_input_scales]);
% scale by taking mean = f(x); we only want to find the location after all
full_alphas = (Kmat\[zeros(1,num_pts);grad'])';
alpha0 = full_alphas(:,1);
alphas = full_alphas(:,2:end);

const_sum = alphas.^2*inv_sqd_input_scales;
local_optimum_flag = const_sum < eps;

x_zoomed = x;
y_zoomed = zeros(num_pts,1);

if all(local_optimum_flag)
    return
end

full_alphas = full_alphas(~local_optimum_flag,:);
alpha0 = alpha0(~local_optimum_flag,:);
alphas = alphas(~local_optimum_flag,:);
const_sum = const_sum(~local_optimum_flag,:);
x = x(~local_optimum_flag,:);
y = y_zoomed(~local_optimum_flag,:);

const_term=sqrt(alpha0.^2+4*const_sum);

% two possible solutions, we'll test them both and return the better
const_left=(-alpha0 - const_term)/(2*const_sum);
const_right=(-alpha0 + const_term)/(2*const_sum);

x_left = const_left*alphas + x;
x_right = const_right*alphas + x;

K = exp(-0.5*(x_left-x).^2*inv_sqd_input_scales);
DK = (x_left-x) .* (K*inv_sqd_input_scales');
mu_left = sum([K, DK].*full_alphas,2);

K = exp(-0.5*(x_right-x).^2*inv_sqd_input_scales);
DK = (x_right-x) .* (K * inv_sqd_input_scales');
mu_right = sum([K, DK].*full_alphas,2);

if strcmpi(min_max_local_optimum_flag,'minimise')
    x = x_right;
    
    left_inds = mu_left < mu_right;
    
    x(left_inds, :) = x_left(left_inds, :); 
    y(left_inds, :) = mu_left(left_inds, :);
elseif strcmpi(min_max_local_optimum_flag,'maximise')
    x = x_right;
    
    left_inds = mu_left < mu_right;
    
    x(left_inds, :) = x_left(left_inds, :); 
    y(left_inds, :) = mu_left(left_inds, :);
end

x_zoomed(~local_optimum_flag, :) = x;
y_zoomed(~local_optimum_flag) = y;