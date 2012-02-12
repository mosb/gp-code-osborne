function stack = sqd_dist_stack(X1, X2)

[N1, D] = size(X1);
[N2, D] = size(X2);

stack = bsxfun(@minus,...
                reshape(X1,N1,1,D),...
                reshape(X2,1,N2,D))...
                .^2;  