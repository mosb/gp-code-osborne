function compile_all_results( outdir, plotdir, tabledir )
% Main script to produce all figures.
%
% outdir: The directory to look in for all the results.
% plotdir: The directory to put all the pplots.

if nargin < 1; outdir = '~/large_results/sbq_results/'; end
if nargin < 2; plotdir = '~/Dropbox/code/papers/sbq-paper/figures/plots/'; end
if nargin < 2; tabledir = '~/Dropbox/code/papers/sbq-paper/tables/'; end

fprintf('Compiling all results...\n');
addpath(genpath(pwd))
mkdir(plotdir);
mkdir(tabledir);

% Get experimental configuration from the definition scripts.
problems = define_integration_problems();
methods = define_integration_methods();
sample_sizes = define_sample_sizes();

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
        for s_ix = 1:num_sample_sizes
            for r = 1:num_repititions
                try
                    % Load one results file.
                    % These are written in run_one_experiment.m.
                    filename = run_one_experiment( problems{p_ix}, methods{m_ix}, sample_sizes(s_ix), r, outdir, true );
                    results = load( filename );

                    % Now save all relevant results into tables.

                    timing_table(m_ix, p_ix, s_ix, r) = results.total_time;
                    mean_log_ev_table(m_ix, p_ix, s_ix, r) = results.mean_log_evidence;
                    var_log_ev_table(m_ix, p_ix, s_ix, r) = results.var_log_evidence;
                    true_log_ev_table(m_ix, p_ix, s_ix, r) = results.problem.true_log_evidence;

                    fprintf('O');       % O for OK.
                catch
                    %disp(lasterror);
                    fprintf('X');       % Never even finished.
                    num_missing = num_missing + 1;
                end
            end
            fprintf(' ');
        end
        fprintf('\n');
    end
    fprintf('\n');
end

method_names = cellfun( @(method) method.acronym, methods, 'UniformOutput', false );
problem_names = cellfun( @(problem) problem.name, problems, 'UniformOutput', false );

% Print tables.
print_table( 'time taken (s)', problem_names, method_names, squeeze(timing_table(:,:,end,1))' );
fprintf('\n\n');
print_table( 'mean_log_ev', problem_names, { method_names{:}, 'Truth' }, ...
    [ squeeze(mean_log_ev_table(:,:,end, 1))' true_log_ev_table(1, :, end, 1)' ] );
fprintf('\n\n');
print_table( 'var_log_ev', problem_names, method_names, ...
    squeeze(var_log_ev_table(:,:,end, 1))' );


% Save tables.
latex_table( [tabledir, 'times_taken.tex'], squeeze(timing_table(:,:,end,1))', problem_names, method_names, 'time taken (s)' );

se = bsxfun(@minus, mean_log_ev_table(:,:,5, 1)', true_log_ev_table(1, :, 5, 1)').^2;
latex_table( [tabledir, 'se.tex'], se, problem_names, method_names, 'squared error at 50 samples' );

for p_ix = 1:num_problems
    for m_ix = 1:num_methods
        r = 1;
        s = 5;%num_sample_sizes;
        true_log_evidence = true_log_ev_table( 1, p_ix, s, r );
        mean_prediction = mean_log_ev_table( m_ix, p_ix, s, r );
        var_prediction = var_log_ev_table( m_ix, p_ix, s, r );
        try
            log_liks(m_ix, p_ix) = logmvnpdf(true_log_evidence, mean_prediction, var_prediction);
        catch
            log_liks(m_ix, p_ix) = NaN;
        end
    end
end
latex_table( [tabledir, 'truth_prob.tex'], -log_liks', problem_names, method_names, 'neg density of truth at 50 samples' );


% Draw some plots
% ================================
close all;

color(1, 1:3) = [1 0.1 0.1];  % red
color(2, 1:3) = [0.1 1 0.1];  % green
color(3, 1:3) = [0.1 0.1 1];  % blue
color(4, 1:3) = [.4 .4 0.1]; 

opacity = 0.1;
edgecolor = 'none';


% Plot log likelihood of true answer, versus number of samples
% ===============================================================
for chosen_problem_ix = 1:num_problems
    cur_problem_name = problem_names{chosen_problem_ix};
    figure; clf;

    try
        for m_ix = 1:num_methods
            for r = 1:num_repititions
                for s = 1:num_sample_sizes
                    true_log_evidence = true_log_ev_table( 1, chosen_problem_ix, s, r );
                    mean_prediction = mean_log_ev_table( m_ix, chosen_problem_ix, s, r );
                    var_prediction = var_log_ev_table( m_ix, chosen_problem_ix, s, r );
                    log_liks(s) = logmvnpdf(true_log_evidence, mean_prediction, var_prediction);
                end
                z_handle(m_ix) = plot( sample_sizes, log_liks, '.', 'Color', color( m_ix, 1:3), 'LineWidth', 1); hold on;
            end
        end
        xlabel('Number of samples');
        ylabel('Log Density of True Value');
        title(cur_problem_name);
        legend(z_handle, method_names);
        %ylim([-3 3 ]);

        filename = sprintf('%slog_of_truth_plot_%s.tikz', plotdir, strrep(cur_problem_name, ' ', '_'));
        matlab2tikz( filename, 'height', '\fheight', 'width', '\fwidth', 'showInfo', false, 'showWarnings', false );
        fprintf('\\input{%s}\n', filename);
    catch e
        %e
    end
end


% Plot squared distance to true answer, versus number of samples
% ===============================================================
for chosen_problem_ix = 1:num_problems
    cur_problem_name = problem_names{chosen_problem_ix};
    figure; clf;

    try
        for m_ix = 1:num_methods
            for r = 1:num_repititions
                for s = 1:num_sample_sizes
                    true_log_evidence = true_log_ev_table( 1, chosen_problem_ix, s, r );
                    mean_prediction = mean_log_ev_table( m_ix, chosen_problem_ix, s, r );
                    var_prediction = var_log_ev_table( m_ix, chosen_problem_ix, s, r );
                    squared_error(s) = (true_log_evidence - mean_prediction)^2;
                end
                z_handle(m_ix) = plot( sample_sizes, squared_error, '.', 'Color', color( m_ix, 1:3), 'LineWidth', 1); hold on;
            end
        end
        xlabel('Number of samples');
        ylabel('Squared Distance to True Value');
        title(cur_problem_name);
        legend(z_handle, method_names);
        %ylim([-3 3 ]);

        filename = sprintf('%sse_plot_%s.tikz', plotdir, strrep(cur_problem_name, ' ', '_'));
        matlab2tikz( filename, 'height', '\fheight', 'width', '\fwidth', 'showInfo', false, 'showWarnings', false );
        fprintf('\\input{%s}\n', filename);
    catch e
        %e
    end
end


if 0
% Plot estimated variance, versus MSE.
% ===============================================================
figure; clf;
for chosen_problem_ix = 1:num_problems
    cur_problem_name = problem_names{chosen_problem_ix};
    figure; clf;

    try
        for m_ix = 1:num_methods
            for r = 1:num_repititions
                for s = 1:num_sample_sizes
                    true_log_evidence = true_log_ev_table( 1, chosen_problem_ix, s, r );
                    mean_prediction = mean_log_ev_table( m_ix, chosen_problem_ix, s, r );
                    var_predictions(s) = var_log_ev_table( m_ix, chosen_problem_ix, s, r );
                    mses(s) = (true_log_evidence - mean_prediction)^2;
                end
                z_handle(m_ix) = plot( var_predictions, mses, '.', 'Color', color( m_ix, 1:3), 'LineWidth', 1); hold on;
            end
        end
        xlabel('Estimated Variance of Estimate of Log Evidence');
        ylabel('Actual Mean Squared Error of Esimate');
        title(cur_problem_name);
        legend(z_handle, method_names);
        xlim([0, 0.1]);
        ylim([0, 0.1]);

        filename = sprintf('%sest_var_vs_mse_%s.tikz', plotdir, strrep(cur_problem_name, ' ', '_'));
        matlab2tikz( filename, 'height', '\fheight', 'width', '\fwidth', 'showInfo', false, 'showWarnings', false );
        fprintf('\\input{%s}\n', filename);
    catch e
        e
    end
end    
end
    

if 0 
% Plot estimated logZ, versus number of samples.
% ===============================================================
figure; clf;
chosen_repetition = 1;  % Repetition shouldn't matter for the truth.
for chosen_problem_ix = 1:num_problems
    cur_problem_name = problem_names{chosen_problem_ix};
    figure; clf;

    try
        for m_ix = 1:num_methods
            for r = 1:num_repititions
                for s = 1:num_sample_sizes
                    mean_predictions(s) = mean_log_ev_table( m_ix, chosen_problem_ix, s, r );
                end
                z_handle(m_ix) = plot( sample_sizes, mean_predictions, '.', 'Color', color( m_ix, 1:3), 'LineWidth', 1); hold on;
            end
        end
        true_log_evidence = squeeze(true_log_ev_table( 1, chosen_problem_ix, ...
                                                      :, chosen_repetition ))';
        truth_handle = plot( sample_sizes, true_log_evidence, 'k-', 'LineWidth', 1); hold on;
        
        xlabel('Number of samples');
        ylabel('Estimated Log Evidence');
        title(cur_problem_name);
        legend([z_handle, truth_handle], {method_names{:}, 'True value'} );
        %xlim([0, 0.1]);
        %ylim([0, 0.1]);

        filename = sprintf('%sest_ev_vs_sample_size_%s.tikz', plotdir, strrep(cur_problem_name, ' ', '_'));
        matlab2tikz( filename, 'height', '\fheight', 'width', '\fwidth', 'showInfo', false, 'showWarnings', false );
        fprintf('\\input{%s}\n', filename);
    catch e
        e
    end
end       
end
    
% Plot one example of mean and variance versus number of samples, for one
% repetition, all methods on one problem.
% ===============================================================

chosen_repetition = 1;
for chosen_problem_ix = 1:num_problems
    cur_problem_name = problem_names{chosen_problem_ix};
    figure; clf;
    for m_ix = 1:num_methods
        % Draw transparent part.
        mean_predictions = squeeze(mean_log_ev_table( m_ix, chosen_problem_ix, ...
                                                      :, chosen_repetition ))';
        var_predictions = squeeze(var_log_ev_table( m_ix, chosen_problem_ix, ...
                                                      :, chosen_repetition ))';
        jbfill(sample_sizes, mean_predictions + 2.*sqrt(var_predictions), ...
                             mean_predictions - 2.*sqrt(var_predictions), ...
                             color(m_ix,1:3), edgecolor, 1, opacity); hold on;
        z_handle(m_ix) = plot( sample_sizes, mean_predictions, '-', 'Color', sqrt(color( m_ix, 1:3) ), 'LineWidth', 1); hold on;
    end

    true_log_evidence = squeeze(true_log_ev_table( 1, chosen_problem_ix, ...
                                                      :, chosen_repetition ))';
    truth_handle = plot( sample_sizes, true_log_evidence, 'k-', 'LineWidth', 1); hold on;
    xlabel('Number of samples');
    ylabel('log evidence');
    title(cur_problem_name);
    legend([z_handle, truth_handle], {method_names{:}, 'True value'} );

    filename = sprintf('%svarplot_%s.tikz', plotdir, strrep(cur_problem_name, ' ', '_'));
    fprintf('\\input{%s}\n', filename);
    matlab2tikz( filename, 'height', '\fheight', 'width', '\fwidth', 'showInfo', false, 'showWarnings', false );
end
