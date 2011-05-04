function best_hypersample = disp_hyperparams(gp, best_hypersample)

if nargin<2
    if isfield(gp.hypersamples,'logL')
        [logL, best_ind] = max([gp.hypersamples.logL]);
        best_hypersample = gp.hypersamples(best_ind).hyperparameters;
    else
        best_hypersample = [gp.hyperparams.priorMean];
    end
end

if nargout == 0
fprintf('Maximum likelihood hyperparameters:\n');
cellfun(@(name, value) fprintf('\t%s\t%g\n', name, value), ...
    {gp.hyperparams.name}', ...
    mat2cell2d(best_hypersample',...
    ones(numel(gp.hyperparams),1),1));
end