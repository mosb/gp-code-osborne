function filename = run_one_experiment( problem, method, nsamples, repitition, outdir, skip )

% Run one experiment, testing one method on one problem.
%
% Inputs:
%   problem is a struct containing the integrand and prior.
%   method is an integration method.
%   nsamples is the number of samples to use.
%   repitition is the replication of this experiment.
%   outdir is the location to save results.
%   skip is a test flag; set it to true if you want to simple print the filename
%   which would have been generated for this experiment.
%
% David Duvenaud
% March 2011
% ===========================

% Generate the filename for this fold.
filename = sprintf( '%sproblem_%s__method_%s__samples_%d_reptition_%d.mat', outdir, problem.name, method.uniquename, nsamples, repitition );

% Set skip to true if you want to just find what the filename for this
% experiment would be.
if skip
    return;
end

% Save the text output.
mkdir(outdir);
diary( [filename '.txt' ] );

fprintf('\n\nRunning\n');
fprintf('          Method: %s\n', method.uniquename );
fprintf('         Problem: %s\n', problem.name );
fprintf('     Num Samples: %d\n', nsamples );
fprintf('       Repitiion: %d\n', repitition );
fprintf('     Description: %s\n', problem.description );
fprintf('Output directory: %s\n', outdir );
fprintf(' Output filename: %s\n\n', filename );

%try   
    % Set the random seed depending on the repitition.
    randn('state', repitition);
    rand('twister', repitition);  
    
    opt = method.opt;
    % Set stopping criterion.
    opt.num_samples = nsamples;

    timestamp = now; tic;    % Start timing.
        
    % Run the experiment.
    [mean_log_evidence, var_log_evidence, samples] = ...
        method.function(problem.log_likelihood_fn, problem.prior, opt);
    
    total_time = toc;        % Stop timing

    % Save all the results.        
    save( filename, 'timestamp', 'total_time', ...
          'mean_log_evidence', 'var_log_evidence', 'samples', ...
          'problem', 'method', 'outdir', 'opt', 'repitition' );
      
    fprintf('\nCompleted experiment.\n\n');
    fprintf('True log evidence:  %d\n', problem.true_log_evidence );
    fprintf('Estimated log evidence:  %d\n', mean_log_evidence );
    fprintf('Estimated variance in log evidence:  %d\n', var_log_evidence );
    fprintf('\nTotal time taken in seconds:  %f\n', total_time );
    fprintf('\n\nSaved to %s\n', filename );
% catch
%     err = lasterror
%     msg = err.message
% end
diary off
