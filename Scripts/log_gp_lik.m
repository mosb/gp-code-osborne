function log_l = log_gp_lik(sample, X_data, y_data, gp)

gp.hypersamples(1).hyperparameters = sample;
gp = revise_gp(X_data, y_data, gp, 'overwrite', [], 1);
log_l = gp.hypersamples(1).logL;