clear all
close all
clc

dataLocation = 'E:\ArtificialNeuralNetwork\ABAQUS_V2\SizePara';

cd (dataLocation)
load('inputData.mat');
radData = inputData;
load('outputData.mat');
stress  = outputData;
strain = linspace(0,0.01,201);
stressNew = zeros(size(stress)); ductility = zeros(size(stress,1),1);
for i = 1:size(stress,1)
    stressNew(i,:)  = (smooth(stress(i,:)));
    idx1 = find(stressNew(i,:)<=0.25*max(stressNew(i,:)));
    idx2 = find(stressNew(i,:)==max(stressNew(i,:)));
    idx3 = idx1(find(idx1>idx2,1));
    ductility(i,1) = strain(idx3);
%     stressNew(i,idx3:end) = 0;
end
stress = stressNew;

% outliersIDX = isoutlier(ductility,'percentiles',[20 80]);
% overWriteIDX1   = find(outliersIDX==1);
% outliersIDX = isoutlier(ductility,'percentiles',[10 90]);
% overWriteIDX2   = find(outliersIDX==1);
% outliersIDX = isoutlier(ductility,'percentiles',[5 95]);
% overWriteIDX3   = find(outliersIDX==1);
% radData = [radData; radData(overWriteIDX1,:);...
%     radData(overWriteIDX2,:); radData(overWriteIDX2,:);...
%     radData(overWriteIDX3,:); radData(overWriteIDX3,:); radData(overWriteIDX3,:)];
% stress = [stress; stress(overWriteIDX1,:);...
%     stress(overWriteIDX2,:); stress(overWriteIDX2,:);...
%     stress(overWriteIDX3,:); stress(overWriteIDX3,:); stress(overWriteIDX3,:)];
% 
% figure
% histogram(ductility,20)
% xlabel('Ductility')
% ylabel('Frequency')
% 
% stressNew = zeros(size(stress)); ductility = zeros(size(stress,1),1);
% for i = 1:size(stress,1)
%     stressNew(i,:)  = smooth((stress(i,:)));
%     idx1 = find(stressNew(i,:)<=0.40*max(stressNew(i,:)));
%     idx2 = find(stressNew(i,:)==max(stressNew(i,:)));
%     idx3 = idx1(find(idx1>idx2,1));
%     ductility(i,1) = strain(idx3);
% %     stressNew(i,idx3:end) = 0;
% end
% stress = stressNew;

figure
histogram(ductility,20)
xlabel('Ductility')
ylabel('Frequency')

outliersIDX = isoutlier(ductility,'mean');
removeIDX   = find(outliersIDX==1);
ductility(removeIDX) = [];
radData(removeIDX,:) = [];
stress(removeIDX,:) = [];

figure
histogram(ductility,20)
xlabel('Ductility')
ylabel('Frequency')

numFig = 15;
modelNum = randi([1 size(radData,1)],numFig,1);
stressTrue = stress(modelNum,:);
figure
for i = 1:numFig
%     subplot(3,5,i);
    hold on; box on; grid on;
    plot(strain(1:4:200),(stressTrue(i,1:4:200)),'-','LineWidth',1.5)
    xlabel('Strain')
    ylabel('Stress (MPa)')
    xlim([0 0.01])
    ylim([0 12])
    set(gca,'FontSize',14)
    hAx = gca;             % handle to current axes
    hAx.XAxis.Exponent=0;  % don't use exponent
end

% radData(modelNum,:) = [];
% stress(modelNum,:) = [];
% m = size(radData,1);
% n = size(radData,2);
% level = 5;
% radDataOld = radData; stressOld = stress;
% for k = 1:level
%     radData = [radData; radDataOld + (randi([-1 1],m,n)).*0.001];
%     stress  = [stress; stressOld];
% end
% 
% for ii = 1:size(radData,1)
%     radData(ii,:) = normalize(radData(ii,:), 'range', [0 1]);
% end

frames = [1:1:200];

stressCNN = stress(:,frames);
strainCNN = strain(1,frames);

numObservations = size(stressCNN,1);
numObservationsTrain = floor(0.6*numObservations);
numObservationsValidation = floor(0.30*numObservations);
numObservationsTest = numObservations - numObservationsTrain - numObservationsValidation;

idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:numObservationsTrain+numObservationsValidation);
idxTest = idx(numObservationsTrain+numObservationsValidation+1:end);


% Separate to training and test data
XTrain = radData(idxTrain,:);
XValidation  = radData(idxValidation,:);
XTest  = radData(idxTest,:);

YTrain = stressCNN(idxTrain,:);
YValidation  = stressCNN(idxValidation,:);
YTest  = stressCNN(idxTest,:);

cd 'E:\ArtificialNeuralNetwork\MATLAB_CNN\Sizeparameter'

save('XTrain','XTrain')
save('XValidation','XValidation')
save('XTest','XTest')

save('YTrain','YTrain')
save('YValidation','YValidation')
save('YTest','YTest')

