function p = lndmvn(x, mu, rooti) 
z = rooti' * (x - mu);
p = -(length(x)/2) * log(2 * pi) - 0.5 * (z(:)' * z(:)) + sum(log(diag(rooti)));
end

