function logP = logP_ring_R_2ways(theta, direction, R,K, p_pulse, decay, q)

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
        logP = logP + logP_ring_R_2ways(theta{i}, direction{i}, R, K, p_pulse, decay, q);
    end
else

N = size(theta, 1);



logP = 0;



for count = 1:length(direction)-1
    
    for i = 1:N
        
        [gap, choice_idx] = min([abs(theta(:, count)-theta(i, count)), 2*pi-(abs(theta(:, count)-theta(i, count)))], [], 2);
        position = zeros(N, 1);
        position(choice_idx == 1) = direction(i, count)*sign(theta(choice_idx==1, count)-theta(i, count));
        position(choice_idx == 2) = -direction(i, count)*sign(theta(choice_idx==2, count)-theta(i, count));
%         numint1 = sum(gap<R & direction(:, count) == -direction(i, count))
%         numint2 = sum(gap<R & direction(:, count) == direction(i, count)) - 1 %-1 to remove self
%         
%         sd = numint1*p_pulse(1) + numint2*p_pulse(2);
        sd =0;
        
        for j = 1:N
            if i ~= j && gap(j) < R 
               
                if direction(i, count) == -direction(j, count)
                    sd = sd + p_pulse(1);
                else
                    sd = sd + p_pulse(2);
                
                end
            
            end
        end
        
            
        
        if direction(i, count+1) == direction(i, count)
            logP = logP + log(1 - 1/(1+exp(-sd-q)));
           
        else
            logP = logP + log(1/(1+exp(-sd-q)));
            
        end
       
        
    end
    
end
       
    

end
    
  



