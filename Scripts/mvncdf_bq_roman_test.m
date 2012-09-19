N = 1000;

% create a random 1000-dim covariance matrix with maximum eigenvalue 1
R = rand(N);
Sigma = R' * R;
max_eig = eigs(Sigma, 1);
Sigma = Sigma ./ max_eig;

mu = zeros(N, 1);
l = zeros(N, 1);
u = inf(N, 1);

% in particular, if opt.data is supplied, the locations for the Gaussian
% convolution observations are supplied.
% opt.data(i).m represents the mean of a Gaussian, 
% opt.data(i).m V the diagonal of its diagonal covariance.

% no data test
% =================================================
opt.data = [];

[ m_Z, sd_Z, data ] = mvncdf_bq( l, u, mu, Sigma, opt )

% single datum test
% =================================================
opt.data(1).m = 0.5*ones(N, 1);
opt.data(1).V = 0.05*ones(N, 1);

[ m_Z, sd_Z, data ] = mvncdf_bq( l, u, mu, Sigma, opt )