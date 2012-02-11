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

easy_1d.name = 'easy 1d';
easy_1d.description = 'A trivial function as a sanity check: a 1D Gaussian.';
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
easy_4d.description = 'A trivial function as a sanity check: a 4D isotropic Gaussian.';
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
easy_10d.description = 'A trivial function as a sanity check: a 10D isotropic Gaussian.';
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
easy_20d.description = 'A trivial function as a sanity check: a 10D isotropic Gaussian.';
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

                                 
bumpy_1d.name = 'bumpy 1d';
bumpy_1d.description = ['A sanity check for estimating variance: '...
                              'a highly varying function, hard to estimate.' ...
                              'This function should cause high uncertainty' ];
bumpy_1d.dimension = 1;
bumpy_1d.prior.mean = .9;
bumpy_1d.prior.covariance = 1.1;
bumpy_1d.log_likelihood_fn = @(x) 0.1.*log(sin( 20.*x ) + 1.1 );
bumpy_1d.true_log_evidence = brute_force_integrate_1d(bumpy_1d);


bumpy_1d_exp.name = 'bumpy 1d exp';
bumpy_1d_exp.description = ['A sanity check for estimating variance: '...
                              'exp of a highly varying function, hard to estimate.'...
                              'Should cause high uncertainty' ];
bumpy_1d_exp.dimension = 1;
bumpy_1d_exp.prior.mean = .9;
bumpy_1d_exp.prior.covariance = 1.1;
bumpy_1d_exp.log_likelihood_fn = @(x) 0.1.*sin( 20.*x ); 
bumpy_1d_exp.true_log_evidence = brute_force_integrate_1d(bumpy_1d_exp);

                                 
two_spikes_1d.name = 'two spikes 1d';
two_spikes_1d.description = 'Two widely separated skinny humps, designed to foil MCMC';
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
      
          
two_hills_1d.name = 'two humps 1d';
two_hills_1d.description = 'Two widely separated skinny humps, designed to foil MCMC';
two_hills_1d.dimension = 1;
two_hills_1d.prior.mean = 0;
two_hills_1d.prior.covariance = 10^2;
likelihood.mean1 = -10; likelihood.mean2 = 10;
likelihood.covariance1 = .25; likelihood.covariance2 = .25;
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
two_spikes_4d.description = 'Two widely separated skinny humps, designed to foil MCMC';
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
two_hills_4d.description = 'Two smooth hills';
two_hills_4d.dimension = 4;
two_hills_4d.prior.mean = zeros(1, two_hills_4d.dimension);
two_hills_4d.prior.covariance = diag(ones(two_hills_4d.dimension, 1) .* 10^2);
likelihood.mean1 = -5 .* ones(1, two_hills_4d.dimension);
likelihood.mean2 = 3 .* ones(1, two_hills_4d.dimension);
likelihood.covariance1 = 4 .* diag(ones(two_hills_4d.dimension,1));
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
funnel_2d.description = 'Radford Neal''s funnel plot, designed to be hard for MH';
funnel_2d.dimension = 2;
funnel_2d.prior.mean = zeros(1, funnel_2d.dimension );
funnel_2d.prior.covariance = 25.*diag(ones(funnel_2d.dimension,1));
funnel_2d.log_likelihood_fn = @(x) arrayfun( @(a,b,c)logmvnpdf(a,b,c), zeros(size(x,1),funnel_2d.dimension - 1), x(:,1), exp(x(:,2)));
% This value was gotten by calling = brute_force_integrate_2d(funnel_2d),
% with dx = 0.01.  However I think it's a little bit off.
%funnel_2d.true_log_evidence = -2.1321289250641388690610256;
% This value was gotten by simple_monte_carlo with 10000 examples.
funnel_2d.true_log_evidence = -2.7480;


% Specify problems.
problems = {};
problems{end+1} = easy_1d;
problems{end+1} = bumpy_1d;
problems{end+1} = bumpy_1d_exp;
problems{end+1} = two_spikes_1d;
problems{end+1} = two_hills_1d;
problems{end+1} = funnel_2d;
problems{end+1} = easy_4d;
problems{end+1} = two_spikes_4d;
problems{end+1} = two_hills_4d;
%problems{end+1} = easy_10d;
%problems{end+1} = easy_20d;

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

