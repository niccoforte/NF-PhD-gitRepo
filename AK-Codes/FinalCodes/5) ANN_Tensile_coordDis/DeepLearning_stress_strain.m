
layers = [
    featureInputLayer(size(deltaXY,2),'Normalization', 'rescale-symmetric')
    fullyConnectedLayer(2048)
    batchNormalizationLayer
    reluLayer
    lstmLayer(1024)
    fullyConnectedLayer(1024)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(1024)
    batchNormalizationLayer
    reluLayer
    lstmLayer(512)
    fullyConnectedLayer(512)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(512)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(size(stressCNN,2))
    regressionLayer];

miniBatchSize = 16;

validationFrequency = floor(length(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
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
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,10),...
    'ExecutionEnvironment','auto',...
    'Verbose',1);

net = trainNetwork(XTrain,YTrain,layers,options);

numFig = 9;
modelNum = randi([1 size(XTest,1)],numFig,1);
stressPred = predict(net,XTest(modelNum,:));
stressTrue = YTest(modelNum,:);
RMSE = sqrt(((stressTrue - stressPred)).^2);

fig = figure;
for i = 1:numFig
    subplot(3,3,i);
    hold on; box on; grid on;
    SE1 = trapz(strainCNN,smooth(stressPred(i,:)));
    SE2 = trapz(strainCNN,stressTrue(i,:));
    plot(strainCNN,(stressTrue(i,:)),'LineWidth',1.5)
    plot(strainCNN,(smooth(stressPred(i,:))),'LineWidth',1.5)
%     xlabel('Strain')
%     ylabel('Stress (MPa)')
    legend('True','Pred')
%     title(strcat(['%error SE: ',num2str(100*abs(SE1-SE2)./SE2)]))
    xlim([0 0.006])
    ylim([0 5])
    set(gca,'FontSize',14)
    set(gcf,'color','w')
    hAx = gca;             % handle to current axes
    hAx.XAxis.Exponent=0;  % don't use exponent
end
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Stress (MPa)');
xlabel(han,'Strain');
set(gca,'FontSize',14);

figure
for i = 1:numFig
    hold on; box on; grid on;
    plot(strainCNN,smooth((stressPred(i,:))),'LineWidth',1.5)
    xlabel('Strain')
    ylabel('Stress (MPa)')
    xlim([0 0.006])
    ylim([0 5])
    set(gca,'FontSize',14)
    hAx = gca;             % handle to current axes
    hAx.XAxis.Exponent=0;  % don't use exponent
end

figure
hold on; box on; grid on;
plot(strainCNN',RMSE','LineWidth',1.5)
xlabel('Strain')
ylabel('RMSE validation')
xlim([0 0.006])
set(gca,'FontSize',14)
hAx = gca;             % handle to current axes
hAx.XAxis.Exponent=0;  % don't use exponent

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