clear all
close all
clc

latticeType = 'tri';
material = 'BrittleMaterial';
numData = 3000;
numNode = 467; %FCC = 912, tri = 467, kagome = 2211
nnx = 21;
nny = nnx;

cd 'G:\FractureToughness_ANN\PythonCode'
nodes_per = importINPfile('tri_21X21per.inp',18,numNode+18-1);
nodes_per(:,1) = [];

dataFileLoc = strcat(['G:\FractureToughness_ANN\ABAQUS\',material]);
mFileLoc    = strcat(['G:\FractureToughness_ANN\MATLAB_ANN\',material,'\',latticeType,...
    '_',num2str(nnx),'X',num2str(nnx)]);

cd (dataFileLoc)
inputDataFileName = strcat(['inputData_',material,latticeType]);
outputDataFileName = strcat(['outputData_',material,latticeType]);
load(inputDataFileName)
load(outputDataFileName)
cd (mFileLoc)

figure
histfit(forceMag(:,2),30)
outputData = forceMag(:,2);
xlabel('Fracture Toughness (MPa\surd{mm})')
ylabel('Frequency')
set(gca,'FontSize',16)
set(gcf,'color','w')

perNodalArray = zeros(1,numNode*2);
perNodalArray(1,1:2:end) = nodes_per(:,1);
perNodalArray(1,2:2:end) = nodes_per(:,2);

inputData = inputData - perNodalArray;

numObservations = numData;
numObservationsTrain = floor(0.6*numObservations);
numObservationsValidation = floor(0.30*numObservations);
numObservationsTest = numObservations - numObservationsTrain - numObservationsValidation;

idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:numObservationsTrain+numObservationsValidation);
idxTest = idx(numObservationsTrain+numObservationsValidation+1:end);


% Separate to training and test data
XTrain = inputData(idxTrain,:);
XValidation  = inputData(idxValidation,:);
XTest  = inputData(idxTest,:);

YTrain = outputData(idxTrain,:);
YValidation  = outputData(idxValidation,:);
YTest  = outputData(idxTest,:);

save('XTrain','XTrain')
save('XValidation','XValidation')
save('XTest','XTest')

save('YTrain','YTrain')
save('YValidation','YValidation')
save('YTest','YTest')

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
