clear all
close all
clc

networkDiagrams = 'G:\FractureToughness_ANN\MATLAB_ANN\BrittleMaterial\Kagome_27X27';
dataLocation = 'G:\FractureToughness_ANN\ABAQUS\BrittleMaterial';

cd 'G:\FractureToughness_ANN\PythonCode'
nodes_per = importINPfile('kagome_27X27per.inp',18,2211+18-1);
nodes_per(:,1) = [];

perNodalArray = zeros(1,2211*2);
perNodalArray(1,1:2:end) = nodes_per(:,1);
perNodalArray(1,2:2:end) = nodes_per(:,2);

topNodes = find(nodes_per(:,2)==max(nodes_per(:,2)));
botNodes = find(nodes_per(:,2)==0);
leftNodes = find(nodes_per(:,1)==0);
rightNodes = find(nodes_per(:,1)==max(nodes_per(:,1)));

boundaryNodes = [topNodes;botNodes;leftNodes;rightNodes];
dofsOfBoundaryNodes = [2*boundaryNodes-1;2*boundaryNodes]';

cd (dataLocation)
load('inputData_BrittleMaterialkagome.mat');
inputData = inputData - perNodalArray;
radData = inputData;

len = size(radData,2);

cd(networkDiagrams)
load('neuralnetworkFractureToughness.mat');

lbVal = min(radData(1,:));
ubVal = max(radData(1,:));

lb = ones(1,len)*lbVal;
ub = ones(1,len)*ubVal;

syms z [1 len]

ftANNOptimizedArray = zeros(5,1); optCoords = zeros(len,5);
for i=1:5
    iniCoords = [474 1122 1272 419 966];
    x0 = radData(iniCoords(i),:); %474 1122 1272 419 966
    fun =@(z)netFunction(net,z,dofsOfBoundaryNodes);
    options = optimoptions('simulannealbnd','PlotFcns',...
        {@saplotbestx,@saplotbestf,@saplotx,@saplotf},'MaxIterations',10000);
    [delta, fval] = simulannealbnd(fun,x0,lb,ub,options);
    optCoords(:,i) = delta';
    ftANNOptimized = predict(net,delta)
    ftANNOptimizedArray(i,1) = ftANNOptimized;
end

load('XTrain')
load('XValidation')
load('XTest')

load('YTrain')
load('YValidation')
load('YTest')

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
scatter(ftANNOptimizedArray,ftANNOptimizedArray,sz+20,'^r','filled')
plot([100 200],[100 200],'--k','LineWidth',1.5)
xlim([100 200])
ylim([100 200])
xlabel('Ini. Frac. Tough. (FEA)')
ylabel('Ini. Frac. Tough. (ANN)')
legend('Train','Validated','Test','Location','SouthEast')
set(gca,'FontSize',16)
set(gcf,'color','w')


function y = netFunction(net,x,dofsOfBoundaryNodes)
    x(:,dofsOfBoundaryNodes) = 0;
    yPred = predict(net,x);
    y = double(1/yPred);
end

function nodes = importINPfile(filename, startRow, endRow)

if nargin<=2
    startRow = 18;
    endRow = 9034;
end


formatSpec = '%8s%14s%s%[^\n\r]';


fileID = fopen(filename,'r');


dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end


fclose(fileID);

raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3]
    % Converts text in the input cell array to numbers. Replaced non-numeric text with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % Create a regular expression to detect and remove non-numeric prefixes and suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData(row), regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if numbers.contains(',')
                thousandsRegExp = '^[-/+]*\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'))
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric text to numbers.
            if ~invalidThousandsSeparator
                numbers = textscan(char(strrep(numbers, ',', '')), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch
            raw{row, col} = rawData{row};
        end
    end
end



R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells


nodes = cell2mat(raw);
end
