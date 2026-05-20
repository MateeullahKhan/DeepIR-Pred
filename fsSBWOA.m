%---------------------------------------------------------------------%
%  Spatial-Bound Whale Optimization Algorithm (WOA) source codes demo version       %
%---------------------------------------------------------------------%


%---Inputs-----------------------------------------------------------
% feat     : feature vector ( Instances x Features )
% label    : label vector ( Instances x 1 )
% N        : Number of whales
% max_Iter : Maximum number of iterations
% b        : Constant 

%---Output-----------------------------------------------------------
% sFeat    : Selected features (instances x features)
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%--------------------------------------------------------------------


%% SB Whale Optimization Algorithm
clc, clear, close
% Benchmark data set 
% Feature Selection for weighted features
%% load the feature sets
load Insulin_PSSM_RICLBP_2476;
load Insulin_QLC_2476;
ESM2 = csvread('ESM2_embeddings_2476.csv');
ProtT5 = csvread('ProtT5_embeddings_2476.csv');

trainL = ProtT5(:,1025);
ESM2_train = ESM2(:,1:1280);
ProtT5_train = ProtT5(:,1:1024);
PsePSSM_train = Insulin_PSSM_RICLBP_2476;
PSSM_RICLBP_train = Insulin_PSSM_RICLBP_2476;
QLC_train = Insulin_QLC_2476;

%% serially integrate feature sets
Insulin_training = [PSSM_RICLBP_train, QLC_train, ESM2_train, ProtT5_train];

%% feature Normalization
feat = Standard_Normalization(Insulin_training);

% Set 20% data as validation set
ho = 0.2; 
% Hold-out method
HO = cvpartition(trainL,'HoldOut',ho);

% Parameter setting
N        = 20; 
max_Iter = 200; 

% Spatial-Bound Whale Optimization Algorithm
[sFeat,Sf,Nf,curve] = SBWOA(feat,trainL,N,max_Iter,HO);

% Accuracy
Acc = jKNN(sFeat,trainL,HO); 
fprintf('\n Accuracy: %g %%',Acc);

% Plot convergence curve
plot(1:max_Iter,curve); 
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('WOA'); grid on;

%% Save Files
SBWOAOpt_Insulin_Training.training = sFeat;
SBWOAOpt_Insulin_Training.idx = Sf;
total_feat = size(Sf,2)+1;
sFeat(:,total_feat) = trainL;
save SBWOA_Insulin_Training_RIC_QLC_ESM2_protT5 SBWOAOpt_Insulin_Training;
csvwrite('SBWOA_Insulin_Training_RIC_QLC_ESM2_protT5.csv',sFeat);


