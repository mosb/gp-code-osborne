function [YMean,YSD,gp,closestInd]=predict_spgp(XStar,...
    gp, quad_gp, params)
% [YMean,YSD,gp,closestInd,output]=predict_spgp(XStar,...
%    gp, params, quad_gp)
% num_steps is the number of iterations allowed to the integration machine

if nargin<4 || isempty(params)
    params = struct();
end
if ~isfield(params,'print')
    params.print=true;
end

if ~isfield(gp, 'hypersamples')
% Initialises hypersamples
gp = hyperparams(gp);
end

if params.print
    display('Beginning prediction with GP')
    start_time = cputime;
end

if nargin<3
    [dummyVar,closestInd] = max([gp.hypersamples.logL]);
    [YMean,YVar] = posterior_spgp(XStar,gp,closestInd,'var_not_cov');
    YSD=sqrt(YVar); 
else
    gp.grad_hyperparams = false; 
    if isstruct(quad_gp)
        weights_mat = bq_params(gp, quad_gp);
    else
        weights_mat = quad_gp;
    end
    hs_weights = weights(gp, weights_mat);
    
    YMean = 0;
    YVar = 0;
    for sample=1:numel(gp.hypersamples)   
        [hs_YMean,hs_YVar] = posterior_spgp(XStar,gp,sample,'var_not_cov');
        
        YMean = YMean + hs_weights(sample)*hs_YMean;
        YVar = YVar + hs_weights(sample)*(hs_YVar + hs_YMean.^2);
    end
    YVar = YVar - YMean.^2;
    YSD = sqrt(YVar); 
end


if params.print
    fprintf('Prediction complete in %g seconds\n', cputime-start_time)
end