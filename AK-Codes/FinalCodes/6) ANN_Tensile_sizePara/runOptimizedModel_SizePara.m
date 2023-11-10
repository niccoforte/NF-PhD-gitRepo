% clear all
% close all
clc

matlabFolder   = 'E:\ArtificialNeuralNetwork\MATLAB_CNN\SizeParameter_v2';
inputFiles     = 'E:\ArtificialNeuralNetwork\MATLAB_CNN\Optimization_120X160_sizepara';
optimizedFiles = 'E:\ArtificialNeuralNetwork\MATLAB_CNN\Optimization_120X160_sizepara';

optNum = 5;
cd (inputFiles)

answer = questdlg('Is it Strain Energy Density based Optimization?', ...
	'Save', ...
	'Yes','No','Cancel','Cancel');
% Handle response
switch answer
    case 'Yes'
        load(strcat(['opt-',num2str(optNum),'_SE.mat']));
        fileName = strcat(['FCC_12X16_size-opt-',num2str(optNum),'_SE.inp']);
    case 'No'
        load(strcat(['opt-',num2str(optNum),'.mat']));
        fileName = strcat(['FCC_12X16_size-opt-',num2str(optNum),'.inp']);
    case 'Cancel'
        error('Canceled by user.')
end

for fileNumber       = 1
    fid=fopen(strcat(['FCC_12X16_size-',num2str(fileNumber),'.inp']));
    tline = fgetl(fid);
    tlines = cell(0,1);
    while ischar(tline)
        tlines{end+1,1} = tline;
        tline = fgetl(fid);
    end
    fclose(fid);
end

ID = strfind(tlines,'*Beam Section, elset=edgeElem-');
indexNode = find(not(cellfun('isempty',ID)));

tlinesNew = tlines;

for ij = 1:length(indexNode)
    tlinesNew{indexNode(ij,1)+1} = strcat(['1., ',num2str(deltaOptOriginal(ij,1))]);
end

cd (optimizedFiles)

if isfile(fileName)
     % File exists.
     delete(fileName)
end

diary(fileName)
for i = 1:size(tlinesNew,1)
    TEXT = tlinesNew(i);
    fprintf(char(TEXT));
    fprintf('\n');
end
diary off

run = strcat(['abaqus job=',fileName,' cpus=4 int'])

system(run)

