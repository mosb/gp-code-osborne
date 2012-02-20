function problems = define_integration_problems()
% Define all integration problems, including priors and exact log-evidences.
%
% Returns a cell array of problems, each one containing:
%
% name
% description
% dimension
% prior.mean
% prior.covariance
% log_likelihood_fn
% true_log_evidence (can be nan for unknown)


simple_test_1d.name = 'simple';
simple_test_1d.description = 'Mike''s simple test';
simple_test_1d.dimension = 1;
simple_test_1d.prior.mean = 0;
simple_test_1d.prior.covariance = 1;
r_mean1 = 1; r_sd1 = 1; r_mean2 = 4; r_sd2 = 1;
normf = @(x,m,sd) (2*pi*sd^2)^(-0.5)*exp(-0.5*(x-m).^2/sd^2);
simple_test_1d.log_likelihood_fn = ...
    @(x) log(normf(x,r_mean1,r_sd1)+normf(x,r_mean2,r_sd2));
simple_test_1d.true_log_evidence = ...
   logsumexp([log_volume_between_two_gaussians(simple_test_1d.prior.mean, ...
                                     simple_test_1d.prior.covariance, ...
                                     r_mean1, r_sd1^2); ...
              log_volume_between_two_gaussians(simple_test_1d.prior.mean, ...
                                     simple_test_1d.prior.covariance, ...
                                     r_mean2, r_sd2^2)]);

simple_test_trans_1d.name = 'simple translated';
simple_test_trans_1d.description = 'Mike''s simple test, translated';
simple_test_trans_1d.dimension = 1;
simple_test_trans_1d.prior.mean = 200;
simple_test_trans_1d.prior.covariance = 1;
r_mean1 = 201; r_sd1 = 1; r_mean2 = 204; r_sd2 = 1;
normf = @(x,m,sd) (2*pi*sd^2)^(-0.5)*exp(-0.5*(x-m).^2/sd^2);
simple_test_trans_1d.log_likelihood_fn = ...
    @(x) log(normf(x,r_mean1,r_sd1)+normf(x,r_mean2,r_sd2));
simple_test_trans_1d.true_log_evidence = ...
   logsumexp([log_volume_between_two_gaussians(simple_test_trans_1d.prior.mean, ...
                                     simple_test_trans_1d.prior.covariance, ...
                                     r_mean1, r_sd1^2); ...
              log_volume_between_two_gaussians(simple_test_trans_1d.prior.mean, ...
                                     simple_test_trans_1d.prior.covariance, ...
                                     r_mean2, r_sd2^2)]);
                                 
simple_test_scale_1d.name = 'simple scaled';
simple_test_scale_1d.description = 'Mike''s simple test, stretched';
simple_test_scale_1d.dimension = 1;
simple_test_scale_1d.prior.mean = 0;
simple_test_scale_1d.prior.covariance = 100;
r_mean1 = 10; r_sd1 = 10; r_mean2 = 40; r_sd2 = 10;
normf = @(x,m,sd) (2*pi*sd^2)^(-0.5)*exp(-0.5*(x-m).^2/sd^2)*10;
simple_test_scale_1d.log_likelihood_fn = ...
    @(x) log(normf(x,r_mean1,r_sd1)+normf(x,r_mean2,r_sd2));
simple_test_scale_1d.true_log_evidence = ...
   logsumexp([log_volume_between_two_gaussians(simple_test_scale_1d.prior.mean, ...
                                     simple_test_scale_1d.prior.covariance, ...
                                     r_mean1, r_sd1^2); ...
              log_volume_between_two_gaussians(simple_test_scale_1d.prior.mean, ...
                                     simple_test_scale_1d.prior.covariance, ...
                                     r_mean2, r_sd2^2)]); %+ log(10);                       

easy_1d.name = 'easy 1d';
easy_1d.description = 'A smooth 1D Gaussian.';
easy_1d.dimension = 1;
easy_1d.prior.mean = .9;
easy_1d.prior.covariance = 1.1;
likelihood.mean = .5;
likelihood.covariance = .1;
easy_1d.log_likelihood_fn = ...
    @(x) logmvnpdf( x, likelihood.mean, likelihood.covariance ); 
easy_1d.true_log_evidence = ...
    log_volume_between_two_gaussians(easy_1d.prior.mean, ...
                                     easy_1d.prior.covariance, ...
                                     likelihood.mean, likelihood.covariance);

                                 
easy_4d.name = 'easy 4d';
easy_4d.description = 'A smooth 4D isotropic Gaussian.';
easy_4d.dimension = 4;
easy_4d.prior.mean = .9 .* ones(1, easy_4d.dimension);
easy_4d.prior.covariance = diag(1.1 .* ones(easy_4d.dimension,1));
likelihood.mean = .5 .* ones(1, easy_4d.dimension);
likelihood.covariance = diag( .25 .* ones(easy_4d.dimension,1));
easy_4d.log_likelihood_fn = ...
    @(x) logmvnpdf( x, likelihood.mean, likelihood.covariance ); 
easy_4d.true_log_evidence = ...
    log_volume_between_two_gaussians(easy_4d.prior.mean, ...
                                     easy_4d.prior.covariance, ...
                                     likelihood.mean, likelihood.covariance);
                                 
                                 
easy_10d.name = 'easy 10d';
easy_10d.description = 'A smooth 10D isotropic Gaussian.';
easy_10d.dimension = 10;
easy_10d.prior.mean = .9 .* ones(1, easy_10d.dimension);
easy_10d.prior.covariance = diag(1.1 .* ones(easy_10d.dimension,1));
likelihood.mean = .5 .* ones(1, easy_10d.dimension);
likelihood.covariance = diag( .25 .* ones(easy_10d.dimension,1));
easy_10d.log_likelihood_fn = ...
    @(x) logmvnpdf( x, likelihood.mean, likelihood.covariance ); 
easy_10d.true_log_evidence = ...
    log_volume_between_two_gaussians(easy_10d.prior.mean, ...
                                     easy_10d.prior.covariance, ...
                                     likelihood.mean, likelihood.covariance); 
                                 
                                 
easy_20d.name = 'easy 20d';
easy_20d.description = 'A smooth 20D isotropic Gaussian.';
easy_20d.dimension = 20;
easy_20d.prior.mean = .9 .* ones(1, easy_20d.dimension);
easy_20d.prior.covariance = diag(1.1 .* ones(easy_20d.dimension,1));
likelihood.mean = .5 .* ones(1, easy_20d.dimension);
likelihood.covariance = diag( .25 .* ones(easy_20d.dimension,1));
easy_20d.log_likelihood_fn = ...
    @(x) logmvnpdf( x, likelihood.mean, likelihood.covariance ); 
easy_20d.true_log_evidence = ...
    log_volume_between_two_gaussians(easy_20d.prior.mean, ...
                                     easy_20d.prior.covariance, ...
                                     likelihood.mean, likelihood.covariance);                                 

                                 
% bumpy_1d.name = 'bumpy 1d';
% bumpy_1d.description = 'a highly varying function';
% bumpy_1d.dimension = 1;
% bumpy_1d.prior.mean = .9;
% bumpy_1d.prior.covariance = 1.1;
% bumpy_1d.log_likelihood_fn = @(x) 0.1.*log(sin( 20.*x ) + 1.1 );
% bumpy_1d.true_log_evidence = brute_force_integrate_1d(bumpy_1d);

bumpy_1d_exp.name = 'bumpy 1d';
bumpy_1d_exp.description = 'a highly varying function';
bumpy_1d_exp.dimension = 1;
bumpy_1d_exp.prior.mean = .9;
bumpy_1d_exp.prior.covariance = 1.1;
bumpy_1d_exp.log_likelihood_fn = @(x) sin( 10.*x ); 
bumpy_1d_exp.true_log_evidence = brute_force_integrate_1d(bumpy_1d_exp);

                                 
two_spikes_1d.name = 'two spikes 1d';
two_spikes_1d.description = 'Two widely separated skinny humps';
two_spikes_1d.dimension = 1;
two_spikes_1d.prior.mean = 0;
two_spikes_1d.prior.covariance = 10^2;
likelihood.mean1 = -10; likelihood.mean2 = 10;
likelihood.covariance1 = .25; likelihood.covariance2 = .25;
scale_factor = 0.05;  % Rescale so it looks nice for plots.
two_spikes_1d.log_likelihood_fn = ...
    @(x) logsumexp([logmvnpdf( x, likelihood.mean1, likelihood.covariance1 ), ...
                    logmvnpdf( x, likelihood.mean2, likelihood.covariance2 )]')' ...
                    + log(scale_factor).*ones(size(x, 1), 1);
       
two_spikes_1d.true_log_evidence = ...                             
    logsumexp([log_volume_between_two_gaussians(two_spikes_1d.prior.mean, ...
                                     two_spikes_1d.prior.covariance, ...
                                     likelihood.mean1, likelihood.covariance1), ...
              log_volume_between_two_gaussians(two_spikes_1d.prior.mean, ...
                                     two_spikes_1d.prior.covariance, ...
                                     likelihood.mean2, likelihood.covariance2)]) ...
              + log(scale_factor);
      
          
two_hills_1d.name = 'two hills 1d';
two_hills_1d.description = 'Two smooth Gaussians';
two_hills_1d.dimension = 1;
two_hills_1d.prior.mean = 0;
two_hills_1d.prior.covariance = 10^2;
likelihood.mean1 = -10; likelihood.mean2 = 10;
likelihood.covariance1 = 5; likelihood.covariance2 = 5;
scale_factor = 0.05;  % Rescale so it looks nice for plots.
two_hills_1d.log_likelihood_fn = ...
    @(x) logsumexp([logmvnpdf( x, likelihood.mean1, likelihood.covariance1 ), ...
                    logmvnpdf( x, likelihood.mean2, likelihood.covariance2 )]')' ...
                    + log(scale_factor).*ones(size(x, 1), 1);
       
two_hills_1d.true_log_evidence = ...                             
    logsumexp([log_volume_between_two_gaussians(two_hills_1d.prior.mean, ...
                                     two_hills_1d.prior.covariance, ...
                                     likelihood.mean1, likelihood.covariance1), ...
              log_volume_between_two_gaussians(two_hills_1d.prior.mean, ...
                                     two_hills_1d.prior.covariance, ...
                                     likelihood.mean2, likelihood.covariance2)]) ...
              + log(scale_factor);
          
          
two_spikes_4d.name = 'two spikes 4d';
two_spikes_4d.description = 'Two widely separated skinny humps';
two_spikes_4d.dimension = 4;
two_spikes_4d.prior.mean = zeros(1, two_spikes_4d.dimension);
two_spikes_4d.prior.covariance = diag(ones(two_spikes_4d.dimension, 1) .* 10^2);
likelihood.mean1 = -10 .* ones(1, two_spikes_4d.dimension);
likelihood.mean2 = 10 .* ones(1, two_spikes_4d.dimension);
likelihood.covariance1 = 0.4 .* diag(ones(two_spikes_4d.dimension,1));
likelihood.covariance2 = 0.4 .* diag(ones(two_spikes_4d.dimension,1));
two_spikes_4d.log_likelihood_fn = ...
    @(x) logsumexp([logmvnpdf( x, likelihood.mean1, likelihood.covariance1 ), ...
                    logmvnpdf( x, likelihood.mean2, likelihood.covariance2 )]')';
two_spikes_4d.true_log_evidence = ...
    logsumexp([log_volume_between_two_gaussians(two_spikes_4d.prior.mean, ...
                                     two_spikes_4d.prior.covariance, ...
                                     likelihood.mean1, likelihood.covariance1), ...
              log_volume_between_two_gaussians(two_spikes_4d.prior.mean, ...
                                     two_spikes_4d.prior.covariance, ...
                                     likelihood.mean2, likelihood.covariance2)]);                                 

                                 
two_hills_4d.name = 'two hills 4d';
two_hills_4d.description = 'Two smooth Gaussian hills';
two_hills_4d.dimension = 4;
two_hills_4d.prior.mean = zeros(1, two_hills_4d.dimension);
two_hills_4d.prior.covariance = diag(ones(two_hills_4d.dimension, 1) .* 10^2);
likelihood.mean1 = -2 .* ones(1, two_hills_4d.dimension);
likelihood.mean2 = 3 .* ones(1, two_hills_4d.dimension);
likelihood.covariance1 = 3 .* diag(ones(two_hills_4d.dimension,1));
likelihood.covariance2 = 2 .* diag(ones(two_hills_4d.dimension,1));
two_hills_4d.log_likelihood_fn = ...
    @(x) logsumexp([logmvnpdf( x, likelihood.mean1, likelihood.covariance1 ), ...
                    logmvnpdf( x, likelihood.mean2, likelihood.covariance2 )]')';
two_hills_4d.true_log_evidence = ...
    logsumexp([log_volume_between_two_gaussians(two_hills_4d.prior.mean, ...
                                     two_hills_4d.prior.covariance, ...
                                     likelihood.mean1, likelihood.covariance1), ...
              log_volume_between_two_gaussians(two_hills_4d.prior.mean, ...
                                     two_hills_4d.prior.covariance, ...
                                     likelihood.mean2, likelihood.covariance2)]);                                  
                                 
funnel_2d.name = 'funnel 2d';
funnel_2d.description = 'Radford Neal''s funnel plot';
funnel_2d.dimension = 2;
funnel_2d.prior.mean = zeros(1, funnel_2d.dimension );
funnel_2d.prior.covariance = 25.*diag(ones(funnel_2d.dimension,1));
funnel_2d.log_likelihood_fn = @(x) arrayfun( @(a,b,c)logmvnpdf(a,b,c), zeros(size(x,1),funnel_2d.dimension - 1), x(:,1), exp(x(:,2)));
% This value was gotten by calling = brute_force_integrate_2d(funnel_2d),
% with dx = 0.01.  However I think it's a little bit off.
%funnel_2d.true_log_evidence = -2.1321289250641388690610256;
% This value was gotten by simple_monte_carlo with 10000 examples.
funnel_2d.true_log_evidence = -2.7480;



friedman_3d.name = 'friedman 3d';
friedman_3d.description = 'isotropic BMC paper experiment.';
friedman_3d.dimension = 3;
friedman_3d.prior.mean = zeros(1, friedman_3d.dimension);
friedman_3d.prior.covariance = diag(ones(friedman_3d.dimension, 1) .* 4);
friedman_data = load('friedman_data.mat');
friedman_3d.log_likelihood_fn = ...
    @(log_hypers)gp_log_likelihood(log_hypers, ...
                                   friedman_data.X, friedman_data.y, @covSEiso);
friedman_3d.true_log_evidence =  -224.042705593331050;   % Based on 1000000 SMC samples.


friedman_7d.name = 'friedman 7d';
friedman_7d.description = 'BMC paper experiment.';
friedman_7d.dimension = 7;
friedman_7d.prior.mean = zeros(1, friedman_7d.dimension);
friedman_7d.prior.covariance = diag(ones(friedman_7d.dimension, 1) .* 4);
friedman_data = load('friedman_data.mat');
friedman_7d.log_likelihood_fn = ...
    @(log_hypers)gp_log_likelihood(log_hypers, ...
                                   friedman_data.X, friedman_data.y, @covSEard);
friedman_7d.true_log_evidence = -215.846016515331058;   % Based on 1000000 SMC samples.



% Define real prawn problems.
% ==================================
load('sixinputs_downsampled');

% Prawn model priors.
priorrange = [10 4 4 10 0.01];
priormean = [0 0 0 0 -7.4950];
priorvars = diag((priorrange.^2)/12);

real_prawn_6d_mean_field.name = 'real prawn 6d mean field';
real_prawn_6d_mean_field.description = 'Integrating over real prawn minds; mean field model.';
real_prawn_6d_mean_field.dimension = 5;
real_prawn_6d_mean_field.prior.mean = priormean;
real_prawn_6d_mean_field.prior.covariance = priorvars;
real_prawn_6d_mean_field.log_likelihood_fn = ...
    loglike_prawn_gaussian(theta, direction, 1);
real_prawn_6d_mean_field.true_log_evidence = NaN;

real_prawn_6d_markov.name = 'real prawn 6d markov';
real_prawn_6d_markov.description = 'Integrating over real prawn minds; best markovian model.';
real_prawn_6d_markov.dimension = 5;
real_prawn_6d_markov.prior.mean = priormean;
real_prawn_6d_markov.prior.covariance = priorvars;
real_prawn_6d_markov.log_likelihood_fn = ...
    loglike_prawn_gaussian(theta, direction, 5);
real_prawn_6d_markov.true_log_evidence = NaN;

real_prawn_6d_non_markov.name = 'real prawn 6d non-markov';
real_prawn_6d_non_markov.description = 'Integrating over real prawn minds; best non-markovian model.';
real_prawn_6d_non_markov.dimension = 5;
real_prawn_6d_non_markov.prior.mean = priormean;
real_prawn_6d_non_markov.prior.covariance = priorvars;
real_prawn_6d_non_markov.log_likelihood_fn = ...
    loglike_prawn_gaussian(theta, direction, 7);
real_prawn_6d_non_markov.true_log_evidence = NaN;


if 0
load('simulated_mf_data');
% NB: parameters are [inv_logistic(R, 0, pi), p_pulse(1), p_pulse(2), 
% ... inv_logistic(decay, 0, 1), q]

sim_prawn_6d_mean_field.name = 'simulated prawn 6d mean field';
sim_prawn_6d_mean_field.description = 'Integrating over simulated prawn minds; mean field model.';
sim_prawn_6d_mean_field.dimension = 5;
sim_prawn_6d_mean_field.prior.mean = priormean;
sim_prawn_6d_mean_field.prior.covariance = priorvars;
sim_prawn_6d_mean_field.log_likelihood_fn = ...
    loglike_prawn_gaussian(theta, direction, 1);
sim_prawn_6d_mean_field.true_log_evidence = -606.550806182059887;  % 10000 samples of SMC.

sim_prawn_6d_markov.name = 'simulated prawn 6d markov';
sim_prawn_6d_markov.description = 'Integrating over simulated prawn minds; best markovian model.';
sim_prawn_6d_markov.dimension = 5;
sim_prawn_6d_markov.prior.mean = priormean;
sim_prawn_6d_markov.prior.covariance = priorvars;
sim_prawn_6d_markov.log_likelihood_fn = ...
    loglike_prawn_gaussian(theta, direction, 5);
sim_prawn_6d_markov.true_log_evidence = -603.243529869901749;    % 10000 samples of SMC.

sim_prawn_6d_non_markov.name = 'simulated prawn 6d non-markov';
sim_prawn_6d_non_markov.description = 'Integrating over simulated prawn minds; best non-markovian model.';
sim_prawn_6d_non_markov.dimension = 5;
sim_prawn_6d_non_markov.prior.mean = priormean;
sim_prawn_6d_non_markov.prior.covariance = priorvars;
sim_prawn_6d_non_markov.log_likelihood_fn = ...
    loglike_prawn_gaussian(theta, direction, 7);
sim_prawn_6d_non_markov.true_log_evidence = -582.248194343939872;  % 10000 samples of SMC.
end

% Specify problems.
problems = {};

problems{end+1} = simple_test_1d;
problems{end+1} = simple_test_trans_1d;
problems{end+1} = simple_test_scale_1d;
problems{end+1} = easy_1d;
problems{end+1} = bumpy_1d_exp;
problems{end+1} = two_spikes_1d;
problems{end+1} = two_hills_1d;
problems{end+1} = funnel_2d;
problems{end+1} = friedman_3d;
problems{end+1} = easy_4d;
problems{end+1} = two_spikes_4d;
problems{end+1} = two_hills_4d;
problems{end+1} = friedman_7d;

problems{end+1} = real_prawn_6d_mean_field;
problems{end+1} = real_prawn_6d_markov;
problems{end+1} = real_prawn_6d_non_markov;

%problems{end+1} = bumpy_1d;
%problems{end+1} = easy_10d;
%problems{end+1} = easy_20d;

%problems{end+1} = sim_prawn_6d_mean_field;
%problems{end+1} = sim_prawn_6d_markov;
%problems{end+1} = sim_prawn_6d_non_markov;
end

function logZ = brute_force_integrate_1d(problem)
    dx = 0.00001;
    xrange = -20:dx:20;
    logZ = log(sum(...
           exp(problem.log_likelihood_fn(xrange')) ...
           .*mvnpdf(xrange', problem.prior.mean, problem.prior.covariance))...
           .*dx);
end

function logZ = brute_force_integrate_2d(problem)
   
    dx = 0.01;
    xrange = -10:dx:10;
    yrange = -10:dx:10;
    Z = 0;
    for y = yrange
        vals = [xrange', ones(length(xrange),1)];
        Z = Z + sum(...
           exp(problem.log_likelihood_fn(vals)) ...
           .*mvnpdf(vals, problem.prior.mean, problem.prior.covariance));
    end
    logZ = log(Z*dx^2);
end

