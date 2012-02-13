function [ y ] = bound( x, l, u )
% [ y ] = bound( x, l, u )
% returns x, bounded above by u and below by l.

y = max(l, min(u, x));

end

