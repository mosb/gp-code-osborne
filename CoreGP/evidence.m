function [logev,unc] = logevidence(covvy)
% Actually returns log of the evidence


if nargout==2
    uncertainty_reqd=true;
else
    uncertainty_reqd=false;
end

if ((isfield(covvy, 'use_derivatives') && covvy.use_derivatives == ...
						 true) || (isfield(covvy, 'covfn') && nargin(covvy.covfn)~=1))
    derivs=true;
else
    % we can determine the gradient of the covariance wrt hyperparams
    derivs=false;
end

% active -eventually this will have an 'update' facility

covvyout=covvy;

lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;

%allow for possibility of either integer or real hyperparams

TSamples=numel(covvy.hyperparams);
hps=1:TSamples;

widthfrac=0.20;

if derivs


    for hyperparam=hps
        
        IndepSamples=covvy.hyperparams(hyperparam).samples;
        priorMean=covvy.hyperparams(hyperparam).priorMean;
        priorSD=covvy.hyperparams(hyperparam).priorSD;
        
        if ~all(~isnan([IndepSamples;priorMean;priorSD]));       
            % This hyperparameter is a dummy - ignore it
            hps=setdiff(hps,hyperparam);
        end
    end
    
    cholKsL=1;

    NSamplesPrev=1;
    notignored=0;  

    for hyperparam=hps

        IndepSamples=covvy.hyperparams(hyperparam).samples;
        NIndepSamples=covvy.hyperparams(hyperparam).NSamples;
        priorMean=covvy.hyperparams(hyperparam).priorMean;
        priorSD=covvy.hyperparams(hyperparam).priorSD;

        NSamples=NSamplesPrev*NIndepSamples;

        notignored=notignored+1;

        width=widthfrac*separation(IndepSamples);

        KsQ=matrify(@(x,y) normpdf(x,y,width),IndepSamples,IndepSamples);

        DKs=matrify(@(x,y) (x-y)/width^2.*normpdf(x,y,width),IndepSamples,IndepSamples); % the variable you're taking the derivative wrt is negative
        DKsD=matrify(@(x,y) 1/width^2*(1-((x-y)/width).^2).*normpdf(x,y,width),IndepSamples,IndepSamples);

        KsL=[KsQ,DKs;-DKs,DKsD];
        
        
        cholKsL=kron2d(cholKsL,chol(KsL));
        remove=4; % This is not a parameter - equal to 2^2 due to structure of cov matrix
        for ind=1:notignored-1
            cholKsL=downdatechol(cholKsL,(remove-1)*NSamples+1:remove*NSamples);
            remove=remove+1;
        end

  
        
       
        


             ns=normpdf(IndepSamples,priorMean,sqrt(width^2+priorSD^2));

        n_hp_A=normpdf(samples_hp,priorMean,sqrt(width^2+priorSD^2));
        n_hp_B=-width^-2*(priorSD^2*(priorSD^2+width^2)^(-1)-1)*(samples_hp-priorMean);

        n_hp=[repmat(n_hp_A,Nhyperparams+1,1);n_hp_C];
        n_hp(inds,:)=n_hp_A.*n_hp_B;
        % then a rearrange

        n=n.*n_hp;
        
        if uncertainty_reqd
        end

        NSamplesPrev=NSamples;
    end
   

    covvyout.KSinv_NS_KSinv=(KSinv_NS/cholKsL)/cholKsL';
    
else
    
    nT_invK=1;
    unc=1;
    widths=nan(max(hps),1);
    priorSDs=nan(max(hps),1);
    for hyperparam=hps
        
        
        type=covvy.hyperparams(hyperparam).type;
        IndepSamples=covvy.hyperparams(hyperparam).samples;
        priorMean=covvy.hyperparams(hyperparam).priorMean;
        priorSD=covvy.hyperparams(hyperparam).priorSD;
        priorSDs(hyperparam)=priorSD;

        if ~all(~isnan([IndepSamples;priorMean;priorSD]));
            % This hyperparameter is a dummy - ignore it
            continue
        end

        width=widthfrac*separation(IndepSamples);
        widths(hyperparam)=width;
    %     fmincon(@(activeTheta) negLogLikelihood(exp(logL),IndepSamples,activeTheta,active,Priorwidth),...
    %         MLwidth(active),[],[],[],[],zeros(length(active),1),largeLimit,[],optimoptions);
        K=matrify(@(x,y) normpdf(x,y,width),IndepSamples,IndepSamples);
        cholK=chol(K);

        switch lower(type)
            case 'real'
             n=normpdf(IndepSamples,priorMean,sqrt(width^2+priorSD^2));
            case 'bounded'
             n=normcdf(IndepSamples,priorMean+priorSD,width)-normcdf(IndepSamples,priorMean-priorSD,width);             
        end
        
        nT_invK_hp=(n'/cholK)/cholK';
        
        nT_invK=kron2d(nT_invK,nT_invK_hp);
        
        if uncertainty_reqd
            unc_hp =-nT_invK_hp*n;
            unc = kron2d(unc,unc_hp);
        end

    end
    
    logLs=[covvy.hypersamples.logL]';
    maxlogL=max(logLs);
    likelihoods=exp(logLs-maxlogL);
    logev=log(nT_invK*likelihoods)+maxlogL;

    unc=unc+(det(2*pi*diag(widths.^2+priorSDs.^2)))^(-0.5);

end


function s = separation(ls) 
if length(ls)<=1
    s=1;
else
    s=(max(ls)-min(ls))/(length(ls)-1);
end
