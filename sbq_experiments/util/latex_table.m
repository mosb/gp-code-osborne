% Turns an array of results into a nice LaTeX table.
% This version also print error bars.
%
% Adapted from Ryan Turner's code.
%
% David Duvenaud
% Feb 2012
% ========================
function latex_table(filename, results, methodNames, metricNames, experimentName)

file = fopen( filename, 'w');

for i = 1:length(methodNames)
    methodNames{i} = strrep(methodNames{i}, '_', ' ');
end

for i = 1:length(metricNames)
    metricNames{i} = strrep(metricNames{i}, '_', ' ');
end

% This is maximum digits left of the dot AFTER shifting the column over by
% the exponent from fixMatrix(). Note that MaxLeftDigits is not the same as
% maxSigFigs.
maxLeftDigits = 3;
maxClip = 10 ^ maxLeftDigits;

[methods metrics] = size(results);
assert(length(methodNames) == methods);
assert(length(metricNames) == metrics);
maxClipCol = zeros(metrics, 1);

% argmin might have trouble if methods is singleton. TODO fix
[best, best_ix] = min(results');
% nearbest keeps track of if we are significantly different from the best
% model.
nearbest = zeros( size(results));
for ii = 1:methods  
    for jj = 1:metrics
        % run a paired ttest
        %if abs(meanScore(ii,jj) - best(jj)) <= stds(ii,jj) + stds(best_ix(jj), jj)
        %h = ttest(results(ii,jj,:), results(best_ix(jj), jj, :));
        if best_ix(ii) == jj %isnan(h) || h == 0
            nearbest(ii,jj) = 1;
        end
    end
end
% Crop the error bars to two digits, shift everything to right exponent, and
% crop the scores to match the error bars.
%[meanScore, errorBar, exponent, prec] = fixMatrix(meanScore, errorBar);

% Print all the usual table header stuff
fprintf(file, '%% --- Automatically generated by latex_table.m ---\n');
fprintf(file, '%% Exported at %s\n', datestr(now()));
fprintf(file, '\\begin{table}[h!]\n');
fprintf(file, '\\caption{{\\small\n');
fprintf(file, '%s\n', experimentName);
fprintf(file, '}}\n');
fprintf(file, '\\label{tbl:%s}\n', experimentName);
fprintf(file, '\\begin{center}\n');
fprintf(file, '\\begin{tabular}{l %s}\n', repmat(' r', 1, metrics));

% first line
fprintf(file, 'Integrand');
for ii = 1:metrics
  fprintf(file, ' & \\rotatebox{0}{ %s } ', metricNames{ii});
end

for jj = 1:methods
  % We don't want the clip to be so small even the best method gets clipped
  orderMagBest = exp10(2+ceil(log10(max(min(results(jj, :)'), 0))));
  maxClipCol(jj) = max(maxClip, orderMagBest);
end
%fprintf(file, ' \\\\ \\hline\n');
fprintf(file, ' \\\\ \\midrule\n');   % Using booktabs


% for each method
for ii = 1:methods
  fprintf(file, methodNames{ii});
  for jj = 1:metrics
    printFormat = ['%4.3f'];
    
    %if best(jj) == ii
    if nearbest(ii, jj)
      %fprintf(file, [' & $\\mathbf{' printFormat '} \\pm %2.1f$'], ...
      fprintf(file, [' & $\\mathbf{' printFormat '}$'], ...
        results(ii, jj));
    elseif results(ii, jj) > maxClipCol(jj)
      fprintf(file, ' & $>$ %d', maxClipCol(jj));
    else
      %fprintf(file, [' & $' printFormat ' \\pm %2.1f$' ], ...
      fprintf(file, [' & $' printFormat '$' ], ...
        results(ii, jj));
    end
  end
  fprintf(file, ' \\\\\n');
end

fprintf(file, '\\end{tabular}\n');
fprintf(file, '\\end{center}\n');
fprintf(file, '\\end{table}\n');
fprintf(file, '%% End automatically generated LaTeX\n');
fclose(file);

% Print something that you can cut and paste into latex
%fprintf('\\input{%s}', filename );
