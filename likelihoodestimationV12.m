function [param,hessian,loglik,s,p,exitflag]=likelihoodestimationV12(MODEL,OBS,SAMPLING,COV1,COV2,initial)
% Fixed Effect on Capture Probability
% OBS: Matrix of Number of Presence (r x c x d)
% SAMPLING: Matrix of Number of SAMPLING (r x c x d)
% COV1 (r x c x d) - Covariate
% COV2 (r x c x d) - Covariate
% MODEL: Model Structure for COVARIATE [X1,X2,X3,X4,X5,X6,X7] 
%                   X1=linear effect on COV 1
%                   X2=second oder on COV1 
%                   X3=third order on COV 1
%                   X4=linear effect on COV2
%                   X5=second order on COV 2
%                   X6=third order on COV 2
%                   X7=interaction term
%       (0=absent; 1=present) 
%% Organize Data
[r,c,d]=size(OBS);  % Year, Month, Location
NOB=SAMPLING-OBS; % Number of no observation
%% Setting options for minimization routine
OPTIONS = optimset(@fminunc);
OPTIONS.MaxFunEvals=10000;
OPTIONS.MaxIter=10000;
OPTIONS.TolFun=1.0000e-21;
OPTIONS.TolX=1.0000e-21;
OPTIONS.Algorithm='quasi-newton';
%%Optimization
[param,fval,exitflag,output,grad,hessian] = fminunc  (@(initial)neg_log_likelihood(initial,OBS,NOB,r,c,d,MODEL,COV1, COV2),initial,OPTIONS);
loglik=-fval; % Log Likelihood
NP=sum(MODEL);
% *************************
% **** COPY Begin ******
% This part needs to be the same as parameters in likelihood.
% *************************
param2=zeros(8+c,1); %10+c-1
param2(1:2)=param(1:2); % Parameters for "p" (alpha and beta) and constant term in "s"
param2(find(MODEL)+2)=param(3:(2+NP)); % Parameters for covariates
param2(10:8+c)=param(3+NP:end); % 
%% Probability of Observation
% Gamma-Binomial Distribution for Probability of Observation given Presence
% for each month at each bay of each year
%% Probability of Occupancy
MONTH=[0,param2(10:end)'];
MONTH=repmat(MONTH,[r,1,d]); % Month Effect
s=param2(2) + ...                                                      % Constant
    param2(3)*COV1+param2(4)*COV1.^2+param2(5)*COV1.^3 + ... 
    param2(6)*COV2+param2(7)*COV2.^2+param2(8)*COV2.^3 + ... 
    param2(9)*COV1.*COV2+...
    MONTH;                                                             % Month Effect
s=exp(s)./(1+exp(s));
% *******COPY End **********
p=exp(param(1))/(1+exp(param(2)));
end

function loglik=neg_log_likelihood(param,OBS,NOB,r,c,d,MODEL,COV1,COV2)
% Negative log likelihood NP = sum(MODEL)+2 (param 1 & 2)+c-1 (months)
%% Probability of Observation
NP=sum(MODEL);
% *************************
% **** COPY Begin ******
% *************************
param2=zeros(8+c,1); %10+c-1
param2(1:2)=param(1:2); % Parameters for "p" (alpha and beta) and constant term in "s"
param2(find(MODEL)+3)=param(3:(2+NP)); % Parameters for covariates
param2(10:8+c)=param(3+NP:end); % 
%% Probability of Observation
% Gamma-Binomial Distribution for Probability of Observation given Presence
% for each month at each bay of each year
p=ones(r,c,d).*exp(param2(1))/(1+exp(param2(1))); 
%% Probability of Occupancy
MONTH=[0,param2(10:end)'];
MONTH=repmat(MONTH,[r,1,d]); % Month Effect
s=param2(2) + ...                                                      % Constant
    param2(3)*COV1+param2(4)*COV1.^2+param2(5)*COV1.^3 + ... 
    param2(6)*COV2+param2(7)*COV2.^2+param2(8)*COV2.^3 + ... 
    param2(9)*COV1.*COV2+...
    MONTH;                                                             % Month Effect
s=exp(s)./(1+exp(s));
% *******COPY End **********
%% Log LIkelihood
LIK=log(((p.^OBS).*((1-p).^NOB).*s+(OBS==0).*(1-s)));
LIK(1:4,:,1)=0; % No sampling in Sabine Lake from 1982 to 1985
loglik=-sum(sum(sum(LIK)));
end