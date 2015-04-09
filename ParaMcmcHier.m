function [out] = ParaMcmcHier(Data,Prior,Mcmc)
% Michael A. Cohen, PhD
% W: www.macohen.net
% E: michael@macohen.net
% Proper citation is appreciated, please cite as:
% Cohen, M. A. (2015). Aysmptotically Exact Embarassingly Parallel MCMC [Computer software]. 
% Retrieved from http://www.macohen.net/software or https://github.com/mcohen05007/parallelmcmc

%% Unpack Estimation Arguments

%MCMC params
    R = Mcmc.R;
    keep = Mcmc.keep;  
    s = Mcmc.s;
    M = Mcmc.M;
    method = Mcmc.method;
    
%Data
    DataM = shard(Data,M,'choice');
    k = size(DataM{1}{1}.X,2);
   
%Prior
    % Density Functions
    loglike = Prior.loglike;
    logpriorden = Prior.logpriorden;
    % Parameters
    deltabar = Prior.deltabar;
    Ad = Prior.Ad;
    nu = Prior.nu;
    V = Prior.V;
    nd = 1;


%% ML estimation of pooled model
x_0 = zeros(k,1);
NU = length(Data);
options = optimset('Display','off','MaxIter',10000,'Algorithm','interior-point','LargeScale','off');
ypool = [];
Xpool = [];
for i=1:NU
    Xpool = [Xpool;Data{i}.X];
    ypool = [ypool;Data{i}.y];
end
[mlepool,~,flag,~,~, H] = fminunc(@(x) -loglike(x,ypool,Xpool),x_0,options);
rootH = eye(k)/chol(H);

%% Run Mcmc in parallel on each shard
out1 = cell(1,M);
parfor m = 1:M
NUm = length(DataM{m});
MLE = zeros(NUm,k);
Hess = zeros(k,k,NUm);
Oll = zeros(1,NUm);
w = 0.1;
for i = 1:NUm
    y = DataM{m}{i}.y;
    X = DataM{m}{i}.X;  
    wgt = size(y,1)/size(ypool,1);
    [mle,ll,flag,~,~, H] = fminunc(@(x) -llmnlFract(loglike,x, y, X, mlepool, rootH, w, wgt),x_0,options);
    [~,pd]=chol(H)
    if flag == 1 && pd==0
        MLE(i,:) = mle';
        Hess(:,:,i) = H;
        Oll(i) = ll;
    else
        MLE(i,:) = zeros(1,k);
        Hess(:,:,i) = eye(k);
        Oll(i) = loglike(zeros(k,1),y,X);        
    end
end

%% Pre-Allocate Store Draws
betadraw  = zeros(R,k);
rootpidraw = zeros(R,k*(k+1)/2);
Otheta = MLE
Z = ones(size(Otheta,1),1);
naccept = 0;
tic
disp('MCMC Iteration Round 1 (Estimated time to end)')
for rep = 1:R
    %% Draw location and scale parameter from from first stage prior
    Hdata = struct('Y',Otheta,'X',Z);
    hier_out = bmreg(Hdata,deltabar,Ad,nu,V,NUm,nd,k);
    betabar = hier_out.beta';  
    rootpi = chol(eye(k)/hier_out.Sigma);
    %% New draw
    for i = 1:NUm
        ROOT = eye(k)/chol(rootpi*rootpi' + Hess(:,:,i));
        [otheta,oll,naccept] = submetrop(loglike,DataM{m}{i}.y,DataM{m}{i}.X,logpriorden,Otheta(i,:)',Oll(i),s,ROOT,betabar,rootpi,naccept,M); 
        Otheta(i,:) = otheta';
        Oll(i) = oll;
    end    
    %% Store draws
    if (mod(rep,keep) == 0) 
        mkeep = rep/keep;
        betadraw(mkeep,:) = betabar';
        rootpidraw(mkeep,:) = rootpi(rootpi~=0);
    end
    %% Compute remaining time
    if (mod(rep,1000) == 0)
        timetoend = (toc/rep) * (R + 1 - rep);
        hours = floor(timetoend/60/60);
        mins = floor((timetoend/60)-hours*60);
        secs = floor(((timetoend/60)-floor(timetoend/60))*60);
        disp(['    ','Worker','    ',num2str(m),'    ','Draw','    ',num2str(rep),'          ',num2str(hours),' ', 'Hours',' ',num2str(mins),' ', 'Minutes',' ',num2str(secs),' ','Seconds'])   
    end   
end
disp(['Total Time Elapsed: ', num2str(round(toc/60)),' ','Minutes']) 
out1{m} = struct('betadraw',betadraw,'rootpidraw',rootpidraw,'Oll',Oll,'Otheta',Otheta,'Hess',Hess,'naccept',naccept);
end
%% Combine Sub-samples
vbetadraw = IMG(out1,'betadraw',method);
vrootpidraw = IMG(out1,'rootpidraw',method);

%% Re-draw theta
parfor m = 1:M
    NUm = length(DataM{m});
    Otheta = out1{m}.Otheta;
    Oll = out1{m}.Oll;
    Hess = out1{m}.Hess;
    naccept = 0;
    thetadraw = zeros(R,NUm,k);
    loglikedraw = zeros(R,NUm);
    utri = triu(ones(k));
    rpid = utri;
tic
disp('MCMC Iteration Round 2 (Estimated time to end)')
for rep = 1:R
    %% New draw
    rpid(utri==1) = vrootpidraw(rep,:);
    for i = 1:NUm
        ROOT = eye(k)/chol(rpid*rpid' + Hess(:,:,i));
        [otheta,oll,naccept] = submetrop(loglike,DataM{m}{i}.y,DataM{m}{i}.X,logpriorden,Otheta(i,:)',Oll(i),s,ROOT,vbetadraw(rep,:)',rpid,naccept,M); 
        Otheta(i,:) = otheta';
        Oll(i) = oll;
    end
    %% Store draws
    if (mod(rep,keep) == 0) 
        mkeep = rep/keep;
        thetadraw(mkeep,:,:) = Otheta;
        loglikedraw(mkeep,:) = Oll;
    end
    %% Compute remaining time
    if (mod(rep,1000) == 0)
        timetoend = (toc/rep) * (R + 1 - rep);
        hours = floor(timetoend/60/60);
        mins = floor((timetoend/60)-hours*60);
        secs = floor(((timetoend/60)-floor(timetoend/60))*60);
        disp(['    ','Worker','    ',num2str(m),'    ','Draw','    ',num2str(rep),'          ',num2str(hours),' ', 'Hours',' ',num2str(mins),' ', 'Minutes',' ',num2str(secs),' ','Seconds'])   
    end   
end
disp(['Total Time Elapsed: ', num2str(round(toc/60)),' ','Minutes']) 
out2{m} = struct('thetadraw',thetadraw,'loglikedraw',loglikedraw,'naccept',naccept);
end

%% Pull together draws from each worker
Thetadraw = zeros(R,NU,k);
Loglikedraw = zeros(R,NU);
it1 = 1;
it2 = 0;
for m = 1:M
    it2 = it2 + size(out2{m}.thetadraw,2);
    Thetadraw(:,it1:it2,:) = out2{m}.thetadraw;
    Loglikedraw(:,it1:it2) = out2{m}.loglikedraw;
    it1 = it2+1;
end

out = struct('betadraw',vbetadraw,'rootpidraw',vrootpidraw,'thetadraw',Thetadraw,'loglikedraw',Loglikedraw);
end

function [otheta,oll,naccept] = submetrop(ll,y,X,lprior,otheta,oll,s,root,thetabar,rootpi,naccept,M)
thetac = otheta + s * root'*randn(numel(otheta),1);
cll = ll(thetac,y,X);
ldiff = cll + lprior(thetac,thetabar,rootpi)/M - oll - lprior(otheta,thetabar,rootpi)/M;
if rand(1)<=exp(ldiff)
    otheta = thetac;
    oll = cll;
    naccept = naccept+1;
end
end

function [DataM]=shard(Data,M,model)
NU = length(Data);
%% Break data into shards
if strcmp('choice',model)
    chunk = ceil(NU/M);
    it1 = 1;
    it2 = chunk;
    rs = randsample(NU,NU);
    DataM = cell(1,M);
    for m = 1:M   
        if it2>NU
            DataM{m} = Data(rs(it1:end));
            it1 = it2+1; 
            it2 = it2+chunk;
        else
            DataM{m} = Data(rs(it1:it2));
            it1 = it2+1; 
            it2 = it2+chunk;            
        end
    end
end
end

function phidraw = IMG(out,param,method)
M = length(out);
[R,k] = size(out{1}.(param));
thetam = 0;
Sigi = 0;
for m = 1:M
    Wm = eye(k)/cov(out{m}.(param));
    thetam = thetam + Wm*mean(out{m}.(param))';
    Sigi = Sigi + Wm;
end
Sig = eye(k)/Sigi;
mu = Sig*thetam;
phidraw = zeros(R,k);
t = unidrnd(R,1,M);
if strcmpi(method,'seminon')
    wt = eps;
    for rep = 1:R
        h = rep^(-1/(4+k)); 
        for m = 1:M
            c = t;
            c(m) = unidrnd(R);
            u = rand(1);
            varbarc = 0;
            for i = 1:M
                varbarc = varbarc + out{i}.(param)(c(i),:)/M;
            end
            wc = 1;
            for i = 1:M
                wc = wc*prod(normpdf(out{i}.(param)(c(i),:),varbarc,h));
            end      
            if u<(wc/wt)
                t = c;
                varbart = varbarc;
                wt = wc;           
            end  
        end
        Sigt = eye(k)/(Sigi+(M/h)*eye(k));
        mut = Sigt*((M/h)*eye(k)*varbart'+Sigi*mu);
        phidraw(rep,:) = (mut + chol(Sigt)*randn(k,1))';
    end
elseif strcmpi(method,'semi')
        Wt = eps;
        for rep = 1:R
            h = rep^(-1/(4+k)); 
            for m = 1:M
                c = t;
                c(m) = unidrnd(R);
                u = rand(1);
                varbarc = 0;
                for i = 1:M
                    varbarc = varbarc + out{i}.(param)(c(i),:)/M;
                end
                wc = 1;
                denom = 1;
                for i = 1:M
                    wc = wc*prod(normpdf(out{i}.(param)(c(i),:),varbarc,h));
                    denom = denom*mvnpdf(out{i}.(param)(c(i),:),mean(out{m}.(param)),cov(out{m}.(param)));
                end  
                Wc = wc*mvnpdf(varbarc,mu',Sig+(h/M)*eye(k))/denom;
                if u<(Wc/Wt)
                    t = c;
                    varbart = varbarc;
                    Wt = Wc;           
                end  
            end
            Sigt = eye(k)/(Sigi+(M/h)*eye(k));
            mut = Sigt*((M/h)*eye(k)*varbart'+Sigi*mu);
            phidraw(rep,:) = (mut + chol(Sigt)*randn(k,1))';
        end
elseif strcmpi(method,'non')
    wt = eps;
    for rep = 1:R
        h = rep^(-1/(4+k)); 
        for m = 1:M
            c = t;
            c(m) = unidrnd(R);
            u = rand(1);
            varbarc = 0;
            for i = 1:M
                varbarc = varbarc + out{i}.(param)(c(i),:)/M;
            end
            wc = 1;
            for i = 1:M
                wc = wc*prod(normpdf(out{i}.(param)(c(i),:),varbarc,h));
            end      
            if u<(wc/wt)
                t = c;
                varbart = varbarc;
                wt = wc;           
            end  
        end
        phidraw(rep,:) = varbart + randn(1,k)*sqrt(h^2/M);
    end
else
    thetam = 0;
    Sigi = 0;
    for m = 1:M
        Wm = eye(k)/cov(out{m}.(param));
        thetam = thetam + Wm*out{m}.(param)';
        Sigi = Sigi + Wm;
    end
    phidraw = ((eye(k)/Sigi)*thetam)';
end
end

function f = llmnlFract(ll,mle, y, X, mlepooled, rootH, w, wgt) 
    z = rootH*(mle - mlepooled);
    f = (1 - w) * ll(mle, y, X) + w * wgt *(-0.5 *(z'*z));
end

function out = bmreg(Data,Bbar,A,nu,V,T,k,neq) 
RA = chol(A);
W = [Data.X;RA];
Z = [Data.Y;RA*Bbar];
IR = eye(k)/chol(W'*W);
Btilde = (IR*IR')*W'*Z;
res = Z-W*Btilde;
wdraw = rwishart(nu+T,eye(neq)/(res'*res + V));
beta = Btilde + IR*randn(k,neq)*wdraw.CI';
out = struct('beta',beta,'Sigma',wdraw.IW);
end

function rout=rwishart(nu, V) 
m = size(V,1);
if (m > 1)
    T = diag(sqrt(chi2rnd(nu,m,1)));
    T = tril(ones(m,m),-1).* randn(m) + T;
else 
    T = sqrt(chi2rnd(nu));
end
U = chol(V);
C = T' * U;
CI = eye(m)/C;
W = C'*C;
IW = CI*CI';
rout = struct('C',C,'CI',CI,'W', W,'IW',IW);
end
