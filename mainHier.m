%% Example Script for Aysmptotically Exact Embarassingly Parallel MCMC for Hierarchical Logit.
% Michael A. Cohen, PhD
% W: www.macohen.net
% E: michael@macohen.net
% Proper citation is appreciated for use or adaptation, please cite as:
% Cohen, M. A. (2015). Aysmptotically Exact Embarassingly Parallel MCMC [Computer software]. 
% Retrieved from http://www.macohen.net/software or https://github.com/mcohen05007/parallelmcmc
clear
clc

%%
% Seed Random number geenrator and use the new SIMD-oriented Fast Mersenne
% Twister only for use with MATLAB 2015a or newer
% rng(0,'simdTwister')
rng(100,'twister')

%% Generate Data (Change dgp.m if you want to experiment with the data generation process)
Data = dgp();

%% Specify Priors and MCMC Parameters
Prior = struct('loglike',@llmnl,'logpriorden',@lndmvn,'thetabar',zeros(3,1),'A',eye(3)*0.01);
Prior.deltabar = zeros(1,k);
Prior.Ad = eye(1)*0.01;
Prior.nu = k+3;
Prior.V =10*Prior.nu* eye(k);
Mcmc = struct('R',5e3,'keep',1,'s',2.93/sqrt(3),'M',4,'method','concensus');

%% Concensus Parallel MCMC (If M =1 Runs Standard MCMC)
[out1] = ParaMcmcHier(Data,Prior,Mcmc);

%% Asymptotically Exact MCMC with Nonparametic Density for Indepent Metropolis within Gibbs
Mcmc.method='non';
[out2] = ParaMcmcHier(Data,Prior,Mcmc);

%% Asymptotically Exact MCMC with Semiparametic Density for Indepent Metropolis within Gibbs
Mcmc.method='semi';
[out3] = ParaMcmcHier(Data,Prior,Mcmc);

%% Asymptotically Exact MCMC with Semiparametic Density with Nonparametric Kernel Weighting for Indepent Metropolis within Gibbs
Mcmc.method='seminon';
[out4] = ParaMcmcHier(Data,Prior,Mcmc);

%% First Display maginal posteriors first-stage prior level utility
% parameters as histograms
% figure
% subplot(4,3,1), hist(out1.betadraw(:,1),30)
% subplot(4,3,2), hist(out1.betadraw(:,2),30)
% subplot(4,3,3), hist(out1.betadraw(:,3),30)
% subplot(4,3,4), hist(out2.betadraw(:,1),30)
% subplot(4,3,5), hist(out2.betadraw(:,2),30)
% subplot(4,3,6), hist(out2.betadraw(:,3),30)
% subplot(4,3,7), hist(out3.betadraw(:,1),30)
% subplot(4,3,8), hist(out3.betadraw(:,2),30)
% subplot(4,3,9), hist(out3.betadraw(:,3),30)
% subplot(4,3,10), hist(out4.betadraw(:,1),30)
% subplot(4,3,11), hist(out4.betadraw(:,2),30)
% subplot(4,3,12), hist(out4.betadraw(:,3),30)

%% Second Display maginal posteriors unit-level utility parameters as histograms
% figure
% subplot(4,3,1), hist(mean(out1.thetadraw(:,:,1),1),30)
% subplot(4,3,2), hist(mean(out1.thetadraw(:,:,2),1),30)
% subplot(4,3,3), hist(mean(out1.thetadraw(:,:,3),1),30)
% subplot(4,3,4), hist(mean(out2.thetadraw(:,:,1),1),30)
% subplot(4,3,5), hist(mean(out2.thetadraw(:,:,2),1),30)
% subplot(4,3,6), hist(mean(out2.thetadraw(:,:,3),1),30)
% subplot(4,3,7), hist(mean(out3.thetadraw(:,:,1),1),30)
% subplot(4,3,8), hist(mean(out3.thetadraw(:,:,2),1),30)
% subplot(4,3,9), hist(mean(out3.thetadraw(:,:,3),1),30)
% subplot(4,3,10), hist(mean(out4.thetadraw(:,:,1),1),30)
% subplot(4,3,11), hist(mean(out4.thetadraw(:,:,2),1),30)
% subplot(4,3,12), hist(mean(out4.thetadraw(:,:,3),1),30)

