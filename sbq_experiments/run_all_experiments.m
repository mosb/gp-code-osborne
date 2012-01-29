function run_all_experiments( outdir )

if nargin < 1
    outdir = 'results/';
end
mkdir( outdir );

fprintf('Running all experiments...\n');

addpath(genpath(pwd));
problems = define_integration_problems();
methods = define_integration_methods();

% Run every combination of experiment.
num_problems = length(problems)
num_methods = length(methods)
for p_ix = 1:num_problems
    for m_ix = 1:num_methods
        run_one_experiment( problems{p_ix}, methods{m_ix}, outdir, false );
    end
end

% Might as well compile all results while we're at it.
compile_all_results( outdir );
