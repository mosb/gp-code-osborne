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

sanity_easy_1d.name = 'sanity easy 1d';
sanity_easy_1d.description = 'A trivial function as a sanity check: a 1D Gaussian.';
sanity_easy_1d.dimension = 1;
sanity_easy_1d.prior.mean = .9;
sanity_easy_1d.prior.covariance = 1.1;
likelihood.mean = .5;
likelihood.covariance = .8;
sanity_easy_1d.log_likelihood_fn = ...
    @(x) logmvnpdf( x, likelihood.mean, likelihood.covariance ); 
sanity_easy_1d.true_log_evidence = ...
    log_volume_between_two_gaussians(sanity_easy_1d.prior.mean, ...
                                     sanity_easy_1d.prior.covariance, ...
                                     likelihood.mean, likelihood.covariance);

                                 
sanity_easy_4d.name = 'sanity easy 4d';
sanity_easy_4d.description = 'A trivial function as a sanity check: a 4D isotropic Gaussian.';
sanity_easy_4d.dimension = 4;
sanity_easy_4d.prior.mean = .9 .* ones(1, sanity_easy_4d.dimension);
sanity_easy_4d.prior.covariance = diag(1.1 .* ones(sanity_easy_4d.dimension,1));
likelihood.mean = .5 .* ones(1, sanity_easy_4d.dimension);
likelihood.covariance = diag( .8 .* ones(sanity_easy_4d.dimension,1));
sanity_easy_4d.log_likelihood_fn = ...
    @(x) logmvnpdf( x, likelihood.mean, likelihood.covariance ); 
sanity_easy_4d.true_log_evidence = ...
    log_volume_between_two_gaussians(sanity_easy_4d.prior.mean, ...
                                     sanity_easy_4d.prior.covariance, ...
                                     likelihood.mean, likelihood.covariance);

                                 
sanity_hard_1d.name = 'sanity hard 1d';
sanity_hard_1d.description = ['A sanity check for estimating variance: '...
                              'a highly varying function, hard to estimate.' ...
                              'This function should cause high uncertainty' ];
sanity_hard_1d.dimension = 1;
sanity_hard_1d.prior.mean = .9;
sanity_hard_1d.prior.covariance = 1.1;
sanity_hard_1d.log_likelihood_fn = @(x) log(sin( 100.*x ) + 1.1 );
sanity_hard_1d.true_log_evidence = brute_force_integrate_1d(sanity_hard_1d);


sanity_hard_1d_exp.name = 'sanity hard 1d exp';
sanity_hard_1d_exp.description = ['A sanity check for estimating variance: '...
                              'exp of a highly varying function, hard to estimate.'...
                              'Should cause high uncertainty' ];
sanity_hard_1d_exp.dimension = 1;
sanity_hard_1d_exp.prior.mean = .9;
sanity_hard_1d_exp.prior.covariance = 1.1;
sanity_hard_1d_exp.log_likelihood_fn = @(x) (sin( 100.*x ) ); 
sanity_hard_1d_exp.true_log_evidence = brute_force_integrate_1d(sanity_hard_1d_exp);

                                 
two_humps_1d.name = 'two humps 1d';
two_humps_1d.description = 'Two widely separated skinny humps, designed to foil MCMC';
two_humps_1d.dimension = 1;
two_humps_1d.prior.mean = 0;
two_humps_1d.prior.covariance = 10^2;
likelihood.mean1 = -10; likelihood.mean2 = 10;
likelihood.covariance1 = .01; likelihood.covariance2 = .01;
% Should we be worried about numerical problems here?
two_humps_1d.log_likelihood_fn = ...
    @(x) log(mvnpdf( x, likelihood.mean1, likelihood.covariance1 ) ...
           + mvnpdf( x, likelihood.mean2, likelihood.covariance2 ));
two_humps_1d.true_log_evidence = ...
    log(exp(log_volume_between_two_gaussians(two_humps_1d.prior.mean, ...
                                     two_humps_1d.prior.covariance, ...
                                     likelihood.mean1, likelihood.covariance1)) + ...
    exp(log_volume_between_two_gaussians(two_humps_1d.prior.mean, ...
                                     two_humps_1d.prior.covariance, ...
                                     likelihood.mean2, likelihood.covariance2)));

                                 
two_humps_4d.name = 'two humps 4d';
two_humps_4d.description = 'Two widely separated skinny humps, designed to foil MCMC';
two_humps_4d.dimension = 4;
two_humps_4d.prior.mean = zeros(1, two_humps_4d.dimension);
two_humps_4d.prior.covariance = diag(ones(two_humps_4d.dimension, 1) .* 10^2);
likelihood.mean1 = -10 .* ones(1, two_humps_4d.dimension);
likelihood.mean2 = 10 .* ones(1, two_humps_4d.dimension);
likelihood.covariance1 = 0.01.*diag(ones(two_humps_4d.dimension,1));
likelihood.covariance2 = 0.01.*diag(ones(two_humps_4d.dimension,1));
two_humps_4d.log_likelihood_fn = ...
    @(x) log(mvnpdf( x, likelihood.mean1, likelihood.covariance1 ) ...
           + mvnpdf( x, likelihood.mean2, likelihood.covariance2 ));
two_humps_4d.true_log_evidence = ...
    log(exp(log_volume_between_two_gaussians(two_humps_4d.prior.mean, ...
                                     two_humps_4d.prior.covariance, ...
                                     likelihood.mean1, likelihood.covariance1)) + ...
    exp(log_volume_between_two_gaussians(two_humps_4d.prior.mean, ...
                                     two_humps_4d.prior.covariance, ...
                                     likelihood.mean2, likelihood.covariance2)));
                                 

funnel_2d.name = 'funnel 2d';
funnel_2d.description = 'Radford Neal''s funnel plot, designed to be hard for MH';
funnel_2d.dimension = 2;
funnel_2d.prior.mean = zeros(1, funnel_2d.dimension );
funnel_2d.prior.covariance = 25.*diag(ones(funnel_2d.dimension,1));
funnel_2d.log_likelihood_fn = @(x) arrayfun( @(a,b,c)logmvnpdf(a,b,c), zeros(size(x,1),funnel_2d.dimension - 1), x(:,1), exp(x(:,2)));
funnel_2d.true_log_evidence = NaN;


% Specify problems.
problems = {};
problems{end+1} = sanity_easy_1d;
problems{end+1} = sanity_easy_4d;
problems{end+1} = sanity_hard_1d;
problems{end+1} = sanity_hard_1d_exp;
problems{end+1} = two_humps_1d;
problems{end+1} = two_humps_4d;
problems{end+1} = funnel_2d;

end

function logZ = brute_force_integrate_1d(problem)
    dx = 0.00001;
    xrange = -20:dx:20;
    logZ = log(sum(...
           exp(problem.log_likelihood_fn(xrange')) ...
           .*mvnpdf(xrange', problem.prior.mean, problem.prior.covariance))...
           .*dx);
end

