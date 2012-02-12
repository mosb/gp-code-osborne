function [ gaussian ] = sqdexp2gaussian( sqd_exp )
% convert a structure containing the hyperparameters of a sqd exp
% covariance into another containing the hyperparameters of a gaussian
% covariance

gaussian = sqd_exp;
gaussian.log_output_scale = ...
    2 * sqd_exp.log_output_scale - 0.5 * log(gaussian_mat(0, gaussian));


end

