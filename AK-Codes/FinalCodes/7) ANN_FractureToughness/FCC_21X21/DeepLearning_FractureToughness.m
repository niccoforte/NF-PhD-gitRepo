
layers = [
    featureInputLayer(size(inputData,2),'Normalization', 'rescale-symmetric')
%     fullyConnectedLayer(4096)
%     batchNormalizationLayer
%     reluLayer
    fullyConnectedLayer(2048)
    batchNormalizationLayer
    reluLayer
%     lstmLayer(1024)
    fullyConnectedLayer(1024)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(512)
    batchNormalizationLayer
    reluLayer
%     lstmLayer(512)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
%     lstmLayer(256)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
%     fullyConnectedLayer(32)
%     batchNormalizationLayer
%     reluLayer
%     fullyConnectedLayer(16)
%     batchNormalizationLayer
%     reluLayer
    fullyConnectedLayer(1)
    regressionLayer]; %qlmseRegressionLayerCustom('qlmse')]; %

miniBatchSize = 16;

validationFrequency = floor(length(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',0.0009, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2427, ...
    'LearnRateDropPeriod',18, ...
    'L2Regularization',6.016e-6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,20),...
    'ExecutionEnvironment','auto',...
    'Verbose',1);

net = trainNetwork(XTrain,YTrain,layers,options);

ftANNTrain = predict(net,XTrain);
ftANNValidation = predict(net,XValidation);
ftANNXTest = predict(net,XTest);

ftFEATrain = YTrain;
ftFEAValidation = YValidation;
ftFEAXTest = YTest;

lim = size(XTrain,1);
numTrain = randi([1 lim],round(lim*0.5),1);
lim = size(XValidation,1);
numValidation = randi([1 lim],round(lim*0.5),1);
lim = size(XTest,1);
numTest = randi([1 lim],round(lim*0.5),1);

sz = 50;
figure
hold on; box on; grid on;
scatter(ftFEATrain(numTrain),ftANNTrain(numTrain),sz,'o','filled')
scatter(ftFEAValidation(numValidation),ftANNValidation(numValidation),sz,'o','filled')
scatter(ftFEAXTest(numTest),ftANNXTest(numTest),sz,'o','filled')
plot([70 140],[70 140],'--k','LineWidth',1.5)
xlim([70 140])
ylim([70 140])
xlabel('Reaction Force (FEA)')
ylabel('Reaction Force (ANN)')
legend('Train','Validated','Test','Location','SouthEast')
set(gca,'FontSize',16)
set(gcf,'color','w')


function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationAccuracy > bestValAccuracy
        valLag = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end