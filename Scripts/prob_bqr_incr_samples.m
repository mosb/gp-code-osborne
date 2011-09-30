max_num_samples = 200;
max_trials = 100;



% matlabpool close force
% matlabpool open

prior = struct();
q = struct();
r = struct();
opt = struct();
r_gp = struct();
sample_struct = struct();


        opt.print = false;
        opt.optim_time = 20;
        opt.num_hypersamples = 50;

prior.means = 0;
prior.sds = 1;

q(1).mean = 0.2;
q(1).cov = 0.5;
q(1).weight = -0.2;

q(2).mean = 0;
q(2).cov = 25;
q(2).weight = 10;

q(3).mean = 0.1;
q(3).cov = 0.3;
q(3).weight = -0.5;

q(4).mean = 1.5;
q(4).cov = 0.1;
q(4).weight = 0.8;

r(1).mean = -1;
r(1).cov = 0.25;
r(1).weight = 0.4;

r(2).mean = 0.5;
r(2).cov = 25;
r(2).weight = 0.8;

r(3).mean = 2;
r(3).cov = 0.5;
r(3).weight = 0.2;

exact = predict_exact(q, r, prior);


p_fn = @(x) normpdf(x, prior.means, prior.sds);
r_fn = @(x) sum([r(:).weight].*normpdf(x, [r(:).mean], sqrt([r(:).cov])));
q_fn = @(x) sum([q(:).weight].*normpdf(x, [q(:).mean], sqrt([q(:).cov])));

d_p_fn = @(x) normpdf(x, prior.means, prior.sds) ...
            * (prior.means - x)/prior.sds;
d_r_fn = @(x) sum([r(:).weight].*(...
                normpdf(x, [r(:).mean], sqrt([r(:).cov])) ...
                .* ([r(:).mean] - x)./sqrt([r(:).cov])));
d_q_fn = @(x) sum([q(:).weight].*(...
                normpdf(x, [q(:).mean], sqrt([q(:).cov])) ...
                .* ([q(:).mean] - x)./sqrt([q(:).cov])));
            
            
p_r_fn = @(x) p_fn(x) * r_fn(x);
