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
            mvnpdf(xrange, problem.prior.mean, problem.prior.covariance), 'g', 'LineWidth', 1); hold on;
        h_ll = plot(xrange, exp(problem.log_likelihood_fn(xrange)), 'b', 'LineWidth', 1);
        h_post = plot(xrange, exp(problem.log_likelihood_fn(xrange)) ...
                      .*mvnpdf(xrange, problem.prior.mean, problem.prior.covariance), ...
                      'r', 'LineWidth', 1);
        %legend([h_prior h_ll h_post], {'Prior', 'Likelihood', 'Posterior'});
        title(problem.name);
        
        % remove axes
        set(gca,'ytick',[]);
        set(gca,'xtick',[]);
        set(gca,'yticklabel',[]);
        set(gca,'xticklabel',[]);        
        
        set(gcf,'units','centimeters')
        set(gcf,'Position',[1 1 40 15])
        savepng(gcf, [plotdir problem.name] );
        
        filename = sprintf('plots/%s.tikz', strrep( problem.name, ' ', '_' ));
        matlab2tikz( filename, 'height', '\fheight', 'width', '\fwidth', 'showInfo', false, 'showWarnings', false );
        fprintf('\\input{%s}\n', filename);
    end
end
