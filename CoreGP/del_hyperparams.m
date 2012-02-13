function del_gp_hypers = del_hyperparams(tl_gp_hypers)

del_gp_hypers = tl_gp_hypers;

del_gp_hypers.log_input_scales = tl_gp_hypers.log_input_scales + log(0.5);
del_gp_hypers.log_output_scales = tl_gp_hypers.log_output_scale + log(0.1);