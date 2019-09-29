% Analysis with year and latitude as covariate with a random observation
% probability (a random factor model)
clear
%% load data
load BagSeine.mat
% PRESENCE {k}  - data for species k
%   .OBS: number of observations of the species (35 years x 12 months x 8
%   bays)
%   .SPCODE: TPWD Species Code
%   .SPNAME: Species Name (common)
%   . SP: Scientific Name
%   .TAXA: 1=Fish, 6 = Invertebrates
%   .TNOBS: Total number of observations
%
% SAMPLLING
%   Number of samples (35 years x 12 months x 8 bays)
%
% INDEX 1: Index of fish species to include
%
% INDEX 2: INDEX of invertebrate species to incldue
%

%% Determin the Model
% First three indices are coefficients on first, second, and third order
% polynomials for the first covariate Y
% Second three indices are coefficients on first, second, and third order
% polynomials for the second covariate Z
% THe last index is for the interaction term
MODEL=[1,1,1,1,1,1,0;
    1,1,1,1,1,0,0;
    1,1,1,1,0,0,0;
    1,1,1,0,0,0,0;
    1,1,0,1,1,1,0;
    1,1,0,1,1,0,0;
    1,1,0,1,0,0,0;
    1,1,0,0,0,0,0;
    1,0,0,1,1,1,0;
    1,0,0,1,1,0,0;
    1,0,0,1,0,0,0;
    1,0,0,0,0,0,0;
    0,0,0,1,1,1,0;
    0,0,0,1,1,0,0;
    0,0,0,1,0,0,0;
    0,0,0,0,0,0,0
    1,1,1,1,1,1,1;
    1,1,1,1,1,0,1;
    1,1,1,1,0,0,1;
    1,1,1,0,0,0,1;
    1,1,0,1,1,1,1;
    1,1,0,1,1,0,1;
    1,1,0,1,0,0,1;
    1,1,0,0,0,0,1;
    1,0,0,1,1,1,1;
    1,0,0,1,1,0,1;
    1,0,0,1,0,0,1;
    1,0,0,0,0,0,1;
    0,0,0,1,1,1,1;
    0,0,0,1,1,0,1;
    0,0,0,1,0,0,1;
    0,0,0,0,0,0,1];

%% Determin the Covariate (Replace Y and Z with other covariates)
Y=repmat(zscore([0:34])',[1,12,8]); % Year Effect
Z(1,1,1:8)=zscore([29.52,29.22,28.36,28.20,27.92,27.49,27.17,26.22]);
Z=repmat(Z,[35,12,1]);

%% ANALYSIS for Each Species % INDEX1 = Fish, % INDEX2=Invertebrates
ID=INDEX1;  % Chnage this to INDEX 2 for Invertebrates
for j=1:length(ID) % for different species
    k=ID(j);
    OBS=PRESENCE{k}.OBS;
    MID=sum(squeeze(sum(OBS,3)))>0;
    OBS2=OBS(:,find(MID),:); % Remove Months of No Observation
    SAMPLING2=SAMPLING(:,find(MID),:);
    [r,c,d]=size(OBS2);  % Year, Month, Location
    COV1=Y(:,find(MID),:);
    COV2=Z(:,find(MID),:);
    for m=1:length(MODEL) % For different model
        for h=1:5 % Five repeated analysis for convergence
            initial=rand(3+sum(MODEL(m,:))+c-1,1)-0.5;
            initial(1:2)=sqrt(initial(1:2).^2); %These have to be always positive
            [param,hessian,loglik,s2,p,exitflag]=likelihoodestimationV2(MODEL(m,:),OBS2,SAMPLING2,COV1,COV2,initial);
            s=zeros(35,12,8);
            s(:,find(MID),:)=s2;
            RESULTS{j,m,h}.param=param;
            RESULTS{j,m,h}.SPCODE=PRESENCE{k}.SPCODE;
            RESULTS{j,m,h}.hessian=hessian;
            RESULTS{j,m,h}.loglig=loglik;
            RESULTS{j,m,h}.s=s;
            RESULTS{j,m,h}.p=p;
            RESULTS{j,m,h}.exitflag=exitflag;
            RESULTS{j,m,h}.MODEL=MODEL(m,:);
            RESULTS{j,m,h}.INDEX=k;
            RESULTS{j,m,h}.COV1=COV1;
            RESULTS{j,m,h}.COV2=COV2;
            RESULTS{j,m,h}.MID=MID; % Months with some observation
        end
    end
end
save resultsV2.mat RESULTS
