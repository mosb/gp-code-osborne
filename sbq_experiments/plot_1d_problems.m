function plot_1d_problems( plotdir )

if nargin < 1
    plotdir = 'plots/';
end
mkdir( plotdir );

addpath(genpath(pwd));
problems = define_integration_problems();
num_problems = length(problems);

for p_ix = 1:num_problems
    problem = problems{p_ix};
    if problem.dimension == 1
        figure(p_ix); clf;
        fprintf('Plotting %s...\n', problem.name );
        xrange = linspace(problem.prior.mean - 2*sqrt(problem.prior.covariance), ...
                          problem.prior.mean + 2*sqrt(problem.prior.covariance), ...
                          1000)';
        h_prior = plot(xrange,...
            mvnpdf(xrange, problem.prior.mean, problem.prior.covariance), 'g'); hold on;
        ll_prior = plot(xrange, exp(problem.log_likelihood_fn(xrange)), 'b');
        legend([h_prior ll_prior], {'Prior', 'Likelihood'});
        title(problem.description)
        set(gcf,'units','centimeters')
        set(gcf,'Position',[1 1 40 15])
        savepng(gcf, [plotdir problem.name] );
    end
end
