function covvy = hyperparams(covvy)

% % set up the parallelisation of hyperparams if desired
% if ~isfield(covvy,'parallel')
%     try
%         matlabpool;
%         covvy.parallel = true;
%     catch
%         covvy.parallel = false;
%     end
% end
% 
% if covvy.parallel
%     isOpen = matlabpool('size') > 0;
%     if ~isOpen
%         matlabpool;
%     end
% end



num_hps = numel(covvy.hyperparams);

% I'm going to store these in covvy because we only need to calculate them
% once and they'll be needed a fair bit in bmcparams
% if (~isfield(covvy,'samplesMean'))
	covvy.samplesMean = cat(2, {covvy.hyperparams(:).priorMean});
% end
% 
% if (~isfield(covvy,'samplesSD'))
	covvy.samplesSD = cat(2, {covvy.hyperparams(:).priorSD});
% end

% if (~isfield(covvy,'names'))
% 	for hyperparam = 1:num_hps
% 		names{hyperparam} = covvy.hyperparams(hyperparam).name;
% 	end
% 	covvy.names = names;
% end

if ~isfield(covvy,'active_hp_inds')
    active=[];
    for hyperparam = 1:num_hps
        if covvy.hyperparams(hyperparam).priorSD <=0
            covvy.hyperparams(hyperparam).type = 'inactive';
        end
        if ~strcmpi(covvy.hyperparams(hyperparam).type,'inactive')
            active=[active,hyperparam];
        else
            covvy.hyperparams(hyperparam).NSamples=1;
        end
    end
    covvy.active_hp_inds=active;
end

% Deal out samples according to priors if it has not already been done
for hyperparam = 1:num_hps
    type = covvy.hyperparams(hyperparam).type;
    
    if (~isfield(covvy.hyperparams(hyperparam), 'samples') || ...
			isempty(covvy.hyperparams(hyperparam).samples));
		mean = covvy.hyperparams(hyperparam).priorMean;
		SD = covvy.hyperparams(hyperparam).priorSD;
        NSamples = covvy.hyperparams(hyperparam).NSamples;
        switch type
            case 'bounded'
                covvy.hyperparams(hyperparam).samples = ...				
                    linspacey(mean - 1 * SD, mean + 1 * SD, ...
									NSamples)';
            case 'real'
                covvy.hyperparams(hyperparam).samples = ...				
                    norminv(1/(NSamples+1):1/(NSamples+1):NSamples/(NSamples+1),mean,SD)';
            case 'mixture'
                mixtureWeights = covvy.hyperparams(hyperparam).mixtureWeights;
                if size(weights,1) == 1
                    mixtureWeights = mixtureWeights';
                    covvy.hyperparams(hyperparam).mixtureWeights = mixtureWeights;
                elseif size(weights,1) ~= 1
                    error(['Mixture Weights for hyperparam number',num2str(hyperparam),'have invalid dimension']);
                end
                samples = nan(NSamples,1);
                cdfs = 1/(NSamples+1):1/(NSamples+1):NSamples/(NSamples+1);
                for i=1:NSamples
                    samples(i) = fsolve(@(x) normcdf(x,mean,SD)*mixtureWeights-cdfs(i),0);
                end
                covvy.hyperparams(hyperparam).samples = samples;
            case 'inactive'
                covvy.hyperparams(hyperparam).samples = mean;
        end
    else
        [samplesize1,samplesize2] = size(covvy.hyperparams(hyperparam).samples);
        if samplesize2==1 && samplesize1>=1
            NSamples = samplesize1;
        elseif samplesize2>1 && samplesize1==1
            covvy.hyperparams(hyperparam).samples = covvy.hyperparams(hyperparam).samples';
            NSamples = samplesize2;
        else
            error(['Samples for hyperparam number',num2str(hyperparam),'have invalid dimension']);
        end
        
        covvy.hyperparams(hyperparam).NSamples = NSamples;
    end
    
    
    

end



samples = allcombs({covvy.hyperparams(:).samples});
num_samples = size(samples,1);
samples_cell = mat2cell2d(samples,ones(num_samples,1),num_hps);
[covvy.hypersamples(1:num_samples).hyperparameters] = samples_cell{:};

% for i = 1:num_samples
% 	covvy.hypersamples(i).hyperparameters = samples(i,:);
% end
