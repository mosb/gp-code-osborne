#!/bin/sh
#
# the next line is a "magic" comment that tells codine to use bash
#$ -S /bin/bash
#
# This script should be what is passed to qsub; its job is just to run one matlab job.

/usr/local/apps/matlab/matlabR2009a/bin/matlab -nodisplay -nojvm -logfile "logs/matlab_log_$1_$2_$3_$4_fold.txt" -r "cd '/home/mlg/dkd23/Dropbox/code/gp-code-osborne/'; addpath(genpath(pwd)); ls; call_one_experiment($1, $2, $3, $4, '/home/mlg/dkd23/large_results/fear_sbq_results/'); exit" 
                                                                                                                                                                          #function call_one_experiment(problem_number, method_number, ...
#                             nsamples, repitition, outdir)

