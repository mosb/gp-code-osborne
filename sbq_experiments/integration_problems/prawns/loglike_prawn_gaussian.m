function log_l_pdf = loglike_prawn_gaussian(theta, direction, model_idx)
%
% Find the log-likelihood of a setting of the nuisance parameters in a
% prawn-mind model.
%
% Reformulated from the code from Rich to use Gaussian priors on unrestricted
% domains.
%
% parameters are:
%
% * range of interaction (for spatial models)
%
% * number of neighbours to interact with (topological model)
%
% * interaction strength with prawns travelling in the opposite direction
%
% * interaction strength with prawns travelling in the same direction (for
% these, positive makes you more likely to turn around, negative less
% likely)
% 
% * decay of memory factor per timestep (1 no decay, 0 no memory)
% 
% * intensity of random turning (I basically fixed this based on 1 prawn
% experiments, otherwise it tries to assign far too much to 'randomness')

if nargin < 4; model_idx = 1; end


% downsample inputs, correlation length is ~10 frames
for i = 1:numel(theta)
    theta = theta(:, 1:2:end);
    direction = direction(:, 1:2:end);
end

switch model_idx
    case 0
        log_l_pdf = @(x) logP_ring_null(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 1
        log_l_pdf = @(x) logP_ring_mf(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 2
        log_l_pdf = @(x) logP_ring_topo(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 3
        log_l_pdf = @(x) logP_ring_R(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 4
        log_l_pdf = @(x) logP_ring_R_2ways(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 5
        log_l_pdf = @(x) logP_ring_R_ahead(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 6
        log_l_pdf = @(x) logP_ring_R_ahead2ways(theta, direction, x(1), x(2), x(3:4), x(5), x(6));       
    case 7
        log_l_pdf = @(x) logP_ring_memory(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 8
        log_l_pdf = @(x) logP_ring_memory_2ways(theta, direction, x(1), x(2), x(3:4), x(5), x(6));     
    case 9
        log_l_pdf = @(x) logP_ring_memory_ahead(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 10
        log_l_pdf = @(x) logP_ring_memory_ahead2ways(theta, direction, x(1), x(2), x(3:4), x(5), x(6));   
end
