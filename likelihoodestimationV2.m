function [param,hessian,loglik,s,p,exitflag]=likelihoodestimationV2(MODEL,OBS,SAMPLING,COV1,COV2,initial)
%[param,hessian,loglik,s,p,exitflag]=likelihoodestimationV2(MODEL,OBS,SAMPLING,COV1,COV2,initial)
%Model with Random Effect on Capture Probability and Two Covariates
%
%This function accomodates different models (inclusion and
%exclusion of covariates) and elimination of months of no observation. 

%% Input Parameters
%**********************************************************************************
%OBS: Matrix of number of presences per month (r x c x d); 
%SAMPLING: Matrix of number of sasmpling occasions per month (r x c x d); 
%COV1 (r x c x d) - Covariate 1; 
%COV2 (r x c x d) - Covariate 2; 
%   r: Number of years (35 for this study)
%   c: Number of months (<=12)
%   d: Number of bays (8 for this study)
%Note: for all data above months of no observation should be eliminated
%(i.e. c<=12).; 
%**********************************************************************************
%
%**********************************************************************************
%MODEL: Model Structure for COVARIATE [X1,X2,X3,X4,X5,X6,X7] ; 
%(0=excluded; 1=included) ; 
%                   X1=linear effect of COV 1; 
%                   X2=second oder effect of COV1 ; 
%                   X3=third order effect of COV 1; 
%                   X4=linear effect effect of COV2; 
%                   X5=second order effect of COV 2; 
%                   X6=third order effect of COV 2; 
%                   X7=interaction term; 
%**********************************************************************************
%
%**********************************************************************************
%initial: Initial guess of parameters (the first two parameters should be
%positive values);
%**********************************************************************************

%% Output parameters
%**********************************************************************************
%param: estimated parameters;        
%hessian: estimated hessiam matrix; 
%loglik: estimated loglikelihood; 
%s: estimated occupancy probability ; 
%exitflag: exit flat (see instruction for "fminunc.m"); 
%**********************************************************************************

%% Organize Data
[r,c,d]=size(OBS);         % Year, Month, Location
NOB=SAMPLING-OBS; % Number of no observation

%% Minimization
OPTIONS = optimset(@fminunc);
OPTIONS.MaxFunEvals=10000;
OPTIONS.MaxIter=10000;
OPTIONS.TolFun=1.0000e-15;
OPTIONS.TolX=1.0000e-15;
OPTIONS.Algorithm='quasi-newton';
[param,fval,exitflag,~,~,hessian] = fminunc  (@(initial)neg_log_likelihood(initial,OBS,NOB,r,c,d,MODEL,COV1, COV2),initial,OPTIONS);
loglik=-fval; % Log Likelihood
%% Calculate occupancy and observation probability based on estimated parameters
%This part should match with the equations within the likelihood function
NP=sum(MODEL);
param2=zeros(9+c,1); %10+c-1
param2(1:3)=param(1:3); % Parameters for "p" (alpha and beta) and constant term in "s"
param2(find(MODEL)+3)=param(4:(3+NP)); % Parameters for covariates
param2(11:9+c)=param(4+NP:end); % Parameter for months (some months are eliminated)
param2(1:2)=sqrt(param2(1:2).^2); % alpha and beta, make them positive  
%Probability of Occupancy
MONTH=[0,param2(11:end)'];
MONTH=repmat(MONTH,[r,1,d]); % Month Effect
s=param2(3) + ...                                                      % Constant
    param2(4)*COV1+param2(5)*COV1.^2+param2(6)*COV1.^3 + ... 
    param2(7)*COV2+param2(8)*COV2.^2+param2(9)*COV2.^3 + ... 
    param2(10)*COV1.*COV2+...
    MONTH;                                                             % Month Effect
s=exp(s)./(1+exp(s));
p=param(1)/param(2);
end
%% Likelihood function
function loglik=neg_log_likelihood(param,OBS,NOB,r,c,d,MODEL,COV1,COV2)
%Negative log likelihood 
%The total number of parameters is 10+c-1 where c is the number of months.
%Some of the parameters are set to 0 to develop different models.
%
%***********************************************************************************
%Development of models by setting some parameters to 0.
NP=sum(MODEL); %Number of coefficients on covariates
param2=zeros(9+c,1); %10+c-1
param2(1:3)=param(1:3); % Parameters for "p" (alpha and beta) and constant term in "s"
param2(find(MODEL)+3)=param(4:(3+NP)); % Parameters on covariates
param2(11:9+c)=param(4+NP:end); % Parameters for month effect
param2(1:2)=sqrt(param2(1:2).^2); % alpha and beta are always positive   
%
%***********************************************************************************
%Probability of Observation
%             Gamma-Binomial Distribution for Probability of Observation 
%             (see MacKenzie et al. 2006)
p=gamma(OBS+NOB+1)./gamma(OBS+1)./gamma(NOB+1).*gamma(param2(1)+OBS).*gamma(param2(2)+NOB)./gamma(param2(1)+param2(2)+NOB+OBS)*gamma(param2(1)+param2(2))/gamma(param2(1))/gamma(param2(2));
%
%***********************************************************************************
%Probability of Occupancy
MONTH=[0,param2(11:end)'];
MONTH=repmat(MONTH,[r,1,d]); % Month Effect
s=param2(3) + ...                                                      % Constant
    param2(4)*COV1+param2(5)*COV1.^2+param2(6)*COV1.^3 + ... 
    param2(7)*COV2+param2(8)*COV2.^2+param2(9)*COV2.^3 + ... 
    param2(10)*COV1.*COV2+...
    MONTH;                                                             % Month Effect
s=exp(s)./(1+exp(s));
%
%***********************************************************************************
%Log LIkelihood
LIK=log(p.*s+(OBS==0).*(1-s));
LIK(1:4,:,1)=0; % No sampling in Sabine Lake from 1982 to 1985
loglik=-sum(sum(sum(LIK)));
end