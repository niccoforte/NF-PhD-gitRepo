clear all
close all
clc

startRow = 18;
endRow = 430;
perNodes            = import_input_file('FCC_12X16_per.inp',startRow,endRow);
perNodesCoordinates = table2array(perNodes(:,2:3));
topNodes = find(perNodesCoordinates(:,2)==320);
botNodes = find(perNodesCoordinates(:,2)==0);
leftNodes = find(perNodesCoordinates(:,1)==0);
rightNodes = find(perNodesCoordinates(:,1)==240);

boundaryNodes = [topNodes;botNodes;leftNodes;rightNodes];
dofsOfBoundaryNodes = [2*boundaryNodes-1;2*boundaryNodes]';
perNodesVector = zeros(1,size(perNodesCoordinates,1)*2);
perNodesVector(1:2:end) = perNodesCoordinates(:,1);
perNodesVector(2:2:end) = perNodesCoordinates(:,2);

stress1 = importdata(strcat(['outputStressStrain-10-1-500.mat']));
stress2 = importdata(strcat(['outputStressStrain-20-1-1500.mat']));
stress3 = importdata(strcat(['outputStressStrain-30-1-500.mat']));

stress = [stress2];
strain = linspace(0,0.006,201);

stressNew = zeros(size(stress)); ductility = zeros(size(stress,1),1);
for i = 1:size(stress,1)
    stressNew(i,:)  = ((smooth(smooth(stress(i,:)))));
    idx1 = find(stressNew(i,:)<=0.30*max(stressNew(i,:)));
    idx2 = find(stressNew(i,:)==max(stressNew(i,:)));
    idx3 = idx1(find(idx1>idx2,1));
    ductility(i,1) = strain(idx3);
%     stressNew(i,idx3:end) = 0;
end

stress = stressNew;

deltaXY1 = importdata(strcat(['inputData-',num2str(10),'-frame-',num2str(0),'.mat']));
deltaXY2 = importdata(strcat(['inputData-',num2str(20),'-frame-',num2str(0),'.mat']));
deltaXY3 = importdata(strcat(['inputData-',num2str(30),'-frame-',num2str(0),'.mat']));

deltaXY = [deltaXY2];

deltaXY(:,topNodes*2) = 0;
deltaXY(:,dofsOfBoundaryNodes) = [];

figure
histogram(ductility,20)
xlabel('Ductility')
ylabel('Frequency')

outliersIDX = isoutlier(ductility,'mean');
removeIDX   = find(outliersIDX==1);
ductility(removeIDX) = [];
deltaXY(removeIDX,:) = [];
stress(removeIDX,:) = [];

figure
histogram(ductility,20)
xlabel('Ductility')
ylabel('Frequency')


numFig = 15;
modelNum = randi([1 size(deltaXY,1)],numFig,1);
stressTrue = stress(modelNum,:);

figure
for i = 1:numFig
%     subplot(3,5,i);
    hold on; box on; grid on;
    plot(strain,(stressTrue(i,:)),'LineWidth',1.5)
    xlabel('Strain')
    ylabel('Stress (MPa)')
    xlim([0 0.006])
    ylim([0 5])
    set(gca,'FontSize',14)
    hAx = gca;             % handle to current axes
    hAx.XAxis.Exponent=0;  % don't use exponent
end


frames = [1:1:200];
stressCNN = stress(:,frames);
strainCNN = strain(1,frames);

numObservations = size(stressCNN,1);
numObservationsTrain = floor(0.7*numObservations);
numObservationsValidation = floor(0.15*numObservations);
numObservationsTest = numObservations - numObservationsTrain - numObservationsValidation;

idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:numObservationsTrain+numObservationsValidation);
idxTest = idx(numObservationsTrain+numObservationsValidation+1:end);


% Separate to training and test data
XTrain = deltaXY(idxTrain,:);
XValidation  = deltaXY(idxValidation,:);
XTest  = deltaXY(idxTest,:);

YTrain = stressCNN(idxTrain,:);
YValidation  = stressCNN(idxValidation,:);
YTest  = stressCNN(idxTest,:);




