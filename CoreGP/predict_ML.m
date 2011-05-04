function [YMean,YSD,covvy,closestInd,output]=predict_ML(XStar,XData,YData,covvy,params)
% [YMean,YSD,covvy,closestInd,output]=predict_ML(XStar,XData,YData,...
% num_steps,covvy,params)
% num_steps is the number of iterations allowed to the integration machine

if nargin<5
    params = struct();
end
if ~isfield(params,'print')
    params.print=true;
end
if ~isfield(params,'max_fn_evals')
    params.max_fn_evals = 300;
end

output=[];

if ~isfield(covvy, 'hypersamples')
% Initialises hypersamples
covvy=hyperparams(covvy);
end
active_inds = covvy.active_hp_inds;

if ~isempty(YData) && ~isempty(active_inds)
    
    if params.max_fn_evals>0
        priorMeans=[covvy.hyperparams(active_inds).priorMean];
        priorSDs=[covvy.hyperparams(active_inds).priorSD];
        lower_bound = priorMeans - 3*priorSDs;
        upper_bound = priorMeans + 3*priorSDs;

        % starting_pt =
        % covvy.hypersamples(closestInd).hyperparameters(active_inds);
        %     options = optimset('GradObj','on', 'OutputFcn',@outfun, 'TolX', eps);
        %     [best_hypersample, neg_logL] = fmincon(objective,...
        %         best_hypersample,...
        %         [],[],[],[],...
        %         lower_bound, upper_bound,...
        %         [],options);

        objective = @(a_hypersample) neg_log_likelihood(a_hypersample,...
            active_inds,XData,YData,covvy,params);

        display('Beginning optimisation of hyperparameters')
        tic;

        Problem.f = objective;

        opts.maxevals = params.max_fn_evals;
        opts.showits = params.print;
        bounds = [lower_bound; upper_bound]';

        [neg_logL, best_a_hypersample] = Direct(Problem, bounds, opts);
        best_a_hypersample = best_a_hypersample';

        display('Completed optimisation of hyperparameters')
        toc;
        
        covvy.hypersamples(1).hyperparameters(active_inds) = best_a_hypersample;
    end

    covvy = gpparams(XData,YData,covvy,'overwrite',[],1);
end

[dummyVar,closestInd] = max([covvy.hypersamples.logL]);
if ~isempty(XStar)
    display('Beginning prediction')
    tic;
    
    [YMean,wC] = gpmeancov(XStar,XData,covvy,closestInd,'var_not_cov');
    YSD=sqrt((wC)); 
    
    display('Prediction complete')
    toc;
else
    YMean = [];
    YSD = [];
end



function [neg_logL neg_glogL] = neg_log_likelihood(a_hypersample,...
    active_inds,XData,YData,covvy,params)


want_derivs = nargout>1;
covvy.use_derivatives = want_derivs;

if size(a_hypersample,1)>1
    covvy.hypersamples(1).hyperparameters(active_inds) = a_hypersample';
else
    covvy.hypersamples(1).hyperparameters(active_inds) = a_hypersample;
end

covvy = gpparams(XData,YData,covvy,'overwrite',[],1);
neg_logL = -covvy.hypersamples(1).logL;
if want_derivs
neg_glogL = -[covvy.hypersamples(1).glogL{active_inds}];
end
if params.print
    fprintf('.');
end