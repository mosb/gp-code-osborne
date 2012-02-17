function logP = logP_ring_null(theta, direction, R, K, p_pulse, decay, q)

if numel(p_pulse) == 1
    p_pulse(2) = -p_pulse(1);
    
end
if R < 0 || R > 2.11*pi || decay < 0 || decay > 1
    logP = -inf;
    return;
end

if iscell(theta)
    logP = 0;
    for i = 1:numel(theta)
        logP = logP + logP_ring_null(theta{i}, direction{i}, R, K, p_pulse, decay, q);
    end
else

N = size(theta, 1);



logP = 0;






for count = 1:length(direction)-1
    
    for i = 1:N
        
      
        
            
        
        if direction(i, count+1) == direction(i, count)
            logP = logP + log(1 - 1/(1+exp(-q)));
           
        else
            logP = logP + log(1/(1+exp(-q)));
            
        end
        
    end
    
end
       
    
    
end
    
  




