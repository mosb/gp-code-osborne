function compile_all_results( outdir )
% Main script to produce all figures.

fprintf('Compiling all results...\n');
addpath(genpath(pwd))


if nargin < 1; outdir = 'results/'; end

problems = define_integration_problems();
methods = define_integration_methods();

num_problems = length(problems);
num_methods = length(methods);


timing_table = NaN( num_methods, num_problems);
mean_log_ev_table = NaN( num_methods, num_problems);
var_log_ev_table = NaN( num_methods, num_problems);
true_log_ev_table = NaN( num_methods, num_problems);
num_missing = 0;
for m_ix = 1:num_methods
    for p_ix = 1:num_problems
        try
            filename = run_one_experiment( problems{p_ix}, methods{m_ix}, outdir, true );
            results = load( filename );
            
            % Results contains 'timestamp', 'total_time', ...
            % 'mean_log_evidence', 'var_log_evidence', 'samples', ...
            % 'problem', 'method', 'outdir' );
            
            % Now save all relevant results into tables.
            timing_table(m_ix, p_ix) = results.total_time;
            mean_log_ev_table(m_ix, p_ix) = results.mean_log_evidence;
            var_log_ev_table(m_ix, p_ix) = results.var_log_evidence;
            true_log_ev_table(m_ix, p_ix) = results.problem.true_log_evidence;
            
            fprintf('O');       % O for OK.
        catch
            %disp(lasterror);
            fprintf('X');       % Never even finished.
            num_missing = num_missing + 1;
        end
        
        method_names{m_ix} = func2str(methods{m_ix});
        problem_names{p_ix} = problems{p_ix}.name;
    end
    fprintf('\n');
end


% Print tables.
print_table( 'time taken (s)', problem_names, method_names, timing_table' );
fprintf('\n\n');
print_table( 'mean_log_ev', problem_names, {'Truth', method_names{:} }, ...
    [ true_log_ev_table(1, :)' mean_log_ev_table'] );
%print_table( 'var_log_ev', problem_names, method_names, var_log_ev_table' );

