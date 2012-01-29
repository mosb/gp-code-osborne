function filename = run_one_experiment( problem, method, outdir, skip )

% Run one experiment, testing one method on one problem.
%
% Inputs:
%   problem is a struct containing the integrand and prior.
%   method is an integration method.
%   outdir is the location to save results.
%   skip is a test flag; set it to true if you want to simple print the filename
%   which would have been generated for this experiment.
%
% David Duvenaud
% March 2011
% ===========================

% Generate the filename for this fold.
filename = sprintf( '%s%s_%s.mat', outdir, problem.name, func2str(method) );

% Set skip to true if you want to just find what the filename for this
% experiment would be.
if skip
    return;
end

% Save the text output.
diary( [filename '.txt' ] );

fprintf('\n\nRunning\n');
fprintf('          Method: %s\n', func2str(method) );
fprintf('         Problem: %s\n', problem.name );
fprintf('     Description: %s\n', problem.description );
fprintf('Output directory: %s\n', outdir );
fprintf(' Output filename: %s\n\n', filename );

%try   
    % Reset the random seed.
    randn('state', 0);
    rand('twister', 0);  
    
    timestamp = now; tic;    % Start timing.
    
    % Run the experiment.
    [mean_log_evidence, var_log_evidence, samples] = ...
        method(problem.log_likelihood_fn, problem.prior);
    
    total_time = toc;        % Stop timing

    % Save all the results.        
    save( filename, 'timestamp', 'total_time', ...
          'mean_log_evidence', 'var_log_evidence', 'samples', ...
          'problem', 'method', 'outdir' );
    fprintf('\nCompleted experiment.\n\nSaved to %s\n', filename );
%catch
    %err = lasterror
    %msg = err.message
%end
diary off
