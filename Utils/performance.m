function [RMSE,normed_RMSE]=performance(XStars,YMean,YSD,XReals,YReals,burn_in)

if nargin<6
    burn_in=0;
end

NStars=size(XStars,1);
sqd_diffs=nan(NStars-burn_in,1);
reals=nan(NStars-burn_in,1);

for ind=(burn_in+1):NStars
    XStar=XStars(ind,:);
    
    mat = bsxfun(@minus,XReals,XStar);
    [a,RightRow] = min(sum(mat.^2,2));

    YReal=YReals(RightRow);
    sqd_diffs(ind)=(YMean(ind)-YReal)^2;
    reals(ind)=YReal;
end

mean_sqd_diffs = mean(sqd_diffs(~isnan(sqd_diffs)));
mean_reals = mean(reals(~isnan(reals)));
mean_sqd_reals = mean(reals(~isnan(reals)).^2);

RMSE=sqrt(mean_sqd_diffs);
%score = score/(max(YReals)-min(YReals));

normed_RMSE=10*log10(mean_sqd_diffs/(mean_sqd_reals-mean_reals^2));
