function ll = llmnl(beta, y, X)
n = size(y,1);
j = size(X,1)/n;
eXbeta = exp(reshape(X * beta, j, n));
ll = sum(log(eXbeta(y'==1)'./sum(eXbeta)));
end
