function debug_sbq(problem_number, nsamples, outdir)
% This function is designed to let a shell script start one experiment.
% Thus everything is idexed by integers.
%
% David Duvenaud
% Feb 2012
% =================


% Set defaults.
if nargin < 1; problem_number = 1; end
if nargin < 2; nsamples = 5; end
if nargin < 3; outdir = 'results/'; end

method_number = 4;
repitition = 1;

problems = define_integration_problems();
methods = define_integration_methods();

sbq_debug_method = methods{method_number};
sbq_debug_method.opt.plots = true;
sbq_debug_method.opt.set_ls_var_method = 'laplace';


% Run experiments.
run_one_experiment( problems{problem_number}, sbq_debug_method, ...
                    nsamples, repitition, outdir, 0 );
