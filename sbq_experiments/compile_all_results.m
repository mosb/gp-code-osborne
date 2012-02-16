function compile_all_results( results_dir, paper_dir )
% Main script to produce all figures.
%
% outdir: The directory to look in for all the results.
% plotdir: The directory to put all the pplots.

draw_plots = false;

if nargin < 1; results_dir = '~/large_results/sbq_results/'; end
if nargin < 2; paper_dir = '~/Dropbox/papers/sbq-paper/'; end
plotdirshort = 'figures/plots/';
tabledirshort = 'tables/';
plotdir = [paper_dir plotdirshort];
tabledir = [paper_dir tabledirshort];

min_samples = 20; % The minimum number of examples before we start making plots.

fprintf('Compiling all results...\n');
autocontent_filename = [paper_dir 'autocontent.tex'];
fprintf('All content listed in %s\n', autocontent_filename);
autocontent = fopen(autocontent_filename, 'w');
fprintf(autocontent, ['\\documentclass{article}\n' ...
    '\\usepackage{preamble}\n' ...
    '\\usepackage{morefloats}\n' ...'
    '\\usepackage{pgfplots}\n' ...
    '\\newlength\\fheight\\newlength\\fwidth\n' ...
    '\\begin{document}\n\n' ...
    '\\input{tables/integrands.tex\n}']);
addpath(genpath(pwd))
%if ~exist([pwd '/' plotdir], 'dir'); mkdir(plotdir); end
%if ~exist([pwd '/' tabledir], 'dir'); mkdir(tabledir); end

% Get experimental configuration from the definition scripts.
problems = define_integration_problems();
methods = define_integration_methods();
%sample_sizes = 
sample_sizes = 1:define_sample_sizes();

num_problems = length(problems);
num_methods = length(methods);
num_sample_sizes = length(sample_sizes);
num_repititions = 1;


timing_table = NaN( num_methods, num_problems, num_sample_sizes, num_repititions);
mean_log_ev_table = NaN( num_methods, num_problems, num_sample_sizes, num_repititions);
var_log_ev_table = NaN( num_methods, num_problems, num_sample_sizes, num_repititions);
true_log_ev_table = NaN( num_methods, num_problems, num_sample_sizes, num_repititions);
num_missing = 0;

for p_ix = 1:num_problems
    fprintf('\nCompiling results for %s...\n', problems{p_ix}.name );
    for m_ix = 1:num_methods
        fprintf( '%6s |', methods{m_ix}.acronym);

        for r = 1:num_repititions
            try
                % Load one results file.
                % These are written in run_one_experiment.m.
                filename = run_one_experiment( problems{p_ix}, methods{m_ix}, sample_sizes(end), r, results_dir, true );
                results = load( filename );

                % Now save all relevant results into tables.
                for s_ix = 1:num_sample_sizes
                    timing_table(m_ix, p_ix, s_ix, r) = results.total_time;
                    mean_log_ev_table(m_ix, p_ix, s_ix, r) = results.mean_log_evidences(s_ix);
                    var_log_ev_table(m_ix, p_ix, s_ix, r) = results.var_log_evidences(s_ix);
                    true_log_ev_table(m_ix, p_ix, s_ix, r) = results.problem.true_log_evidence;
                end
                samples{m_ix, p_ix} = results.samples;
                if any(isnan(results.mean_log_evidences(min_samples:end))) ...
                        || any(isnan(results.var_log_evidences((min_samples:end))))
                    fprintf('N');
                else
                    fprintf('O');       % O for OK
                end
            catch
                %disp(lasterror);
                fprintf('X');       % Never even finished.
                num_missing = num_missing + 1;
            end
        end
        fprintf(' ');
        
        fprintf('\n');
    end
    fprintf('\n');
end

% Some sanity checking.
%for p_ix = 1:num_problems
    % Check that the true value for every problem was recorded as being the
    % same for all repititions, timesteps and methods tried.
%    assert(all(all(all(true_log_ev_table(:, p_ix, :, :) == ...
%                       true_log_ev_table(1, p_ix, 1, 1)))));
%end

% Normalize everything.
%for p_ix = 1:num_problems
%    mean_log_ev_table(:, p_ix, :, :) = mean_log_ev_table(:, p_ix, :, :) - true_log_ev_table(:, p_ix, :, :);
    % I think variances should stay the same... need to think about this more.
%end


method_names = cellfun( @(method) method.acronym, methods, 'UniformOutput', false );
problem_names = cellfun( @(problem) problem.name, problems, 'UniformOutput', false );

% Print tables.
print_table( 'time taken (s)', problem_names, method_names, ...
    squeeze(timing_table(:,:,end,1))' );
fprintf('\n\n');
print_table( 'mean_log_ev', problem_names, { method_names{:}, 'Truth' }, ...
    [ squeeze(mean_log_ev_table(:,:,end, 1))' true_log_ev_table(1, :, end, 1)' ] );
fprintf('\n\n');
print_table( 'var_log_ev', problem_names, method_names, ...
    squeeze(var_log_ev_table(:,:,end, 1))' );


% Save tables.
latex_table( [tabledir, 'times_taken.tex'], squeeze(timing_table(:,:,end,1))', ...
    problem_names, method_names, 'time taken (s)' );
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'times_taken.tex']);

se = bsxfun(@minus, mean_log_ev_table(:,:,end, 1)', true_log_ev_table(1, :, end, 1)').^2;
latex_table( [tabledir, 'se.tex'], log(real(se)), problem_names, method_names, ...
    sprintf('log squared error at %i samples', sample_sizes(end)) );
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'se.tex']);

for p_ix = 1:num_problems
    for m_ix = 1:num_methods
        r = 1;
        s = sample_sizes(end);
        true_log_evidence = true_log_ev_table( 1, p_ix, s, r );
        mean_prediction = mean_log_ev_table( m_ix, p_ix, s, r );
        var_prediction = var_log_ev_table( m_ix, p_ix, s, r );
        try
            log_liks(m_ix, p_ix) = logmvnpdf(true_log_evidence, ...
                                             mean_prediction, var_prediction);
        catch
            log_liks(m_ix, p_ix) = NaN;
        end
    end
end
latex_table( [tabledir, 'truth_prob.tex'], -log_liks', problem_names, ...
     method_names, sprintf('neg log density of truth at %i samples', ...
                           sample_sizes(end)) );
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'truth_prob.tex']);


% Draw some plots
% ================================
close all;

color(1, 1:3) = [1   0.1 0.1];  % red
color(2, 1:3) = [0.1   1 0.1];  % green
color(3, 1:3) = [0.1   0.1 1];  % blue
color(4, 1:3) = [.4  0.4 0.1];  % dark yellow
color(5, 1:3) = [0.1   1   1];  % cyan
color(6, 1:3) = [0.9 0.1 0.9];  % purple

opacity = 0.1;
edgecolor = 'none';

if draw_plots

% Print legend.
% =====================
%method_names = cellfun( @(name) strrep( name, '_', ' '), method_names, ...
%                       'UniformOutput', false);
figure; clf;
for m_ix = 1:num_methods
    z_handle(m_ix) = plot( 0, 0, '-', 'Color', sqrt(color( m_ix, :) ), 'LineWidth', 1); hold on;
end
truth_handle = plot( 1, 1, 'k-', 'LineWidth', 1); hold on;
h_l = legend([z_handle, truth_handle], {method_names{:}, 'True value'} );
filename = 'legend';
axis off;
%matlab2tikz( [plotdir filename], 'height', '\fheight', 'width', '\fwidth', 'showInfo', false, 'showWarnings', false );
%fprintf(autocontent, '\n\\begin{figure}\n\\centering\\setlength\\fheight{3cm}\\setlength\\fwidth{3cm}\\input{%s}\n\\end{figure}\n', [plotdirshort filename]);    
set_fig_units_cm( 4, 5 )
matlabfrag([plotdir filename]);
fprintf(autocontent, '\\psfragfig{%s}\n', [plotdirshort filename]);    


label_fontsize = 10;

%zlabel('Z','fontsize',12,'userdata','matlabfrag:$\mathcal Z$')

% Plot log likelihood of true answer, versus number of samples
% ===============================================================
plotted_sample_set = min_samples:num_sample_sizes;
%figure_string = '\n\\begin{figure}\n\\centering\\setlength\\fheight{14cm}\\setlength\\fwidth{12cm}\\input{%s}\n\\end{figure}\n';
figure_string = '\\psfragfig{%s}\n';


for p_ix = 1:num_problems
    cur_problem_name = problem_names{p_ix};
    figure; clf;

    try
        for m_ix = 1:num_methods
            for r = 1:num_repititions
                for s = plotted_sample_set
                    true_log_evidence = true_log_ev_table( 1, p_ix, s, r );
                    mean_prediction = mean_log_ev_table( m_ix, p_ix, s, r );
                    var_prediction = var_log_ev_table( m_ix, p_ix, s, r );
                    neg_log_liks(m_ix, s) = -real(logmvnpdf(true_log_evidence, ...
                                                mean_prediction, var_prediction));
                end
                z_handle(m_ix) = plot( plotted_sample_set, ...
                    real(neg_log_liks(m_ix, plotted_sample_set)), '-', ...
                    'Color', color( m_ix, :), 'LineWidth', 1); hold on;
            end
        end
        
        xlabel('Number of samples', 'fontsize', label_fontsize);
        ylabel('Neg Log Density of True Value', 'fontsize', label_fontsize);
        title(cur_problem_name, 'fontsize', label_fontsize);

        filename = sprintf('log_of_truth_plot_%s', strrep(cur_problem_name, ' ', '_'));

        set_fig_units_cm( 8, 6 );
        matlabfrag([plotdir filename]);
        fprintf(autocontent, figure_string, [plotdirshort filename]);    
    catch e
        %e
    end
end


% Plot log of squared distance to true answer, versus number of samples
% ======================================================================
for p_ix = 1:num_problems
    cur_problem_name = problem_names{p_ix};
    figure; clf;

    try
        for m_ix = 1:num_methods
            for r = 1:num_repititions
                for s = min_samples:num_sample_sizes
                    true_log_evidence = true_log_ev_table( 1, p_ix, s, r );
                    mean_prediction = mean_log_ev_table( m_ix, p_ix, s, r );
                    var_prediction = var_log_ev_table( m_ix, p_ix, s, r );
                    squared_error(s) = (true_log_evidence - mean_prediction)^2;
                end
                z_handle(m_ix) = semilogy( plotted_sample_set, ...
                    squared_error(plotted_sample_set), '-',...
                    'Color', color( m_ix, :), 'LineWidth', 1); hold on;
            end
        end
        xlabel('Number of samples');
        ylabel('Squared Distance to True Value');
        title(cur_problem_name);


        filename = sprintf('se_plot_%s', strrep(cur_problem_name, ' ', '_'));
        set_fig_units_cm( 8, 6 );
        matlabfrag([plotdir filename]);
        fprintf(autocontent, figure_string, [plotdirshort filename]);    
    catch e
        %e
    end
end

  
% Plot one example of mean and variance versus number of samples, for one
% repetition, all methods on one problem.
% ===============================================================

chosen_repetition = 1;
for p_ix = 1:num_problems
    cur_problem_name = problem_names{p_ix};
    figure; clf;
    try
        for m_ix = 1:num_methods
            % Draw transparent part.
            mean_predictions(m_ix, :) = squeeze(mean_log_ev_table( m_ix, p_ix, ...
                                                          :, chosen_repetition ))';
            var_predictions = real(squeeze(var_log_ev_table( m_ix, p_ix, ...
                                                          :, chosen_repetition ))');
            jbfill(plotted_sample_set, mean_predictions(m_ix, plotted_sample_set) + 2.*sqrt(var_predictions(plotted_sample_set)), ...
                                 mean_predictions(m_ix, plotted_sample_set) - 2.*sqrt(var_predictions(plotted_sample_set)), ...
                                 color(m_ix,:), edgecolor, 1, opacity); hold on;
            z_handle(m_ix) = plot( plotted_sample_set, ...
                mean_predictions(m_ix, plotted_sample_set), '-', ...
                'Color', color( m_ix, :), 'LineWidth', 1); hold on;
        end

        true_log_evidence = squeeze(true_log_ev_table( 1, p_ix, ...
                                                          :, chosen_repetition ))';
        truth_handle = plot( plotted_sample_set, ...
            true_log_evidence(plotted_sample_set), 'k-', 'LineWidth', 1); hold on;
        xlabel('Number of samples');
        ylabel('log evidence');
        title(cur_problem_name);
        ylim( [min(min((mean_predictions(:, plotted_sample_set)))), max(max((mean_predictions(:, plotted_sample_set)))) + 1] );
        %legend([z_handle, truth_handle], {method_names{:}, 'True value'} );

        filename = sprintf('varplot_%s', strrep(cur_problem_name, ' ', '_'));
        set_fig_units_cm( 8, 6 );
        matlabfrag([plotdir filename], 'renderer', 'opengl', 'dpi', 200);
        fprintf(autocontent, figure_string, [plotdirshort filename]);    
    catch e
        e
    end
end




% Plot sample paths
% ===============================================================

chosen_repetition = 1;
for p_ix = 1:num_problems
    cur_problem = problems{p_ix};
    figure; clf;
    try
        for m_ix = 1:num_methods
            cur_samples = samples{m_ix, p_ix};
            if isfield(cur_samples, 'locations')
                cur_samples = cur_samples.locations;
            end
            if ~isempty(cur_samples)
%                z_handle(m_ix) = plot( cur_samples(:,1), '.', ...
 %                   'Color', color( m_ix, :), 'LineWidth', 1); hold on;
                % Plot the sample locations.
                start_ix = 1;
                end_ix = length(cur_samples(:,1));
                h_samples = plot3( (start_ix:end_ix)', ...
                   cur_samples(start_ix:end_ix,1), ...
                   zeros( end_ix - start_ix + 1, 1 ), '.', ...
                   'Color', color( m_ix, :));   hold on;      
            end
        end
        
        bounds = ylim;
        xrange = linspace( bounds(1), bounds(2), 1000)';
        n = length(xrange);        
        true_plot_depth = sample_sizes(end);
                
        % Plot the prior.
        h_prior = plot3(repmat(true_plot_depth + 1,n,1), xrange,...
            mvnpdf(xrange, cur_problem.prior.mean(1), cur_problem.prior.covariance(1)), 'k', 'LineWidth', 2); hold on;

        % Plot the likelihood function.
        like_func_vals = ...
            exp(cur_problem.log_likelihood_fn(...
            [xrange zeros(n, cur_problem.dimension - 1)]));
        % Rescale it to match the vertical scale of the prior.
        like_func_vals = like_func_vals ./ max(like_func_vals) ...
            .* mvnpdf(0, 0, cur_problem.prior.covariance(1));        
        h_ll = plot3(repmat(true_plot_depth,n,1), xrange, like_func_vals, 'g', 'LineWidth', 2);
        
        xlabel('Number of samples');
        ylabel('sample location');
        title(cur_problem.name);
        xlim( [ 0 true_plot_depth ] );
        grid on;
        set(gca,'ydir','reverse')
        view(-72, 42);
        
        filename = sprintf('sampleplot_%s', strrep(cur_problem.name, ' ', '_'));
        set_fig_units_cm( 8, 6 );
        matlabfrag([plotdir filename]);
        fprintf(autocontent, figure_string, [plotdirshort filename]);    
    catch e
        e
    end
end
end
fprintf(autocontent, '\n\n\\end{document}');
fclose(autocontent);

%close all;
