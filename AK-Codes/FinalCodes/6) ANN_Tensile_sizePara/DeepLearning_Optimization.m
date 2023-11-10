clear all
close all
clc

networkDiagrams = 'E:\ArtificialNeuralNetwork\MATLAB_CNN\Optimization_120X160_sizepara';
dataLocation = 'E:\ArtificialNeuralNetwork\ABAQUS_V2\SizePara';

cd (dataLocation)
load('inputData.mat');
radData = inputData;
frames = [1:1:200];

len = size(radData,2);
strain = linspace(0,0.01,201);
strain = strain(1,frames);

cd(networkDiagrams)
net = load('networkStressStrain.mat');


lbVal = min(radData(1,:));
ubVal = max(radData(1,:));

lb = ones(1,len)*lbVal;
ub = ones(1,len)*ubVal;

syms z [1 len]
x0 = radData(3506,:);
fun =@(z)netFunction(net,z,strain);
funSE =@(z)netFunctionStrainEnergy(net,z,strain);
options = optimoptions('simulannealbnd','PlotFcns',...
          {@saplotbestx,@saplotbestf,@saplotx,@saplotf},'MaxIterations',10000);
[deltaOpt, fvalDuct] = simulannealbnd(funSE,x0,lb,ub,options);


deltaOptOriginal = deltaOpt';
deltaOpt = deltaOpt(1,:);
fvalDuct = fvalDuct(1,:);


yOptPred = predict(net.net,deltaOpt);
SE_or_ductility = 1/fvalDuct(1)

figure
hold on; box on; grid on;
plot(strain./0.0043,smooth(yOptPred)./10.45,'LineWidth',1.5)
% legend('Perfect True', 'Optimized Pred')
xlabel('Strain')
ylabel('Stress (MPa)')
xlim([0 2])
ylim([0 1.2])
set(gca,'FontSize',14)
set(gcf,'color','w')
hAx = gca;             
hAx.XAxis.Exponent=0;


function y = netFunction(net,x,strain)
    yPred = smooth(predict(net.net,x));
    if max(yPred) >=9.45 && max(yPred)<=9.975
        idx1 = find(yPred<=0.25*max(yPred));
        idx2 = find(yPred==max(yPred));
        ductility = strain(idx1(find(idx1>max(idx2),1)));
    else
        ductility = 1e-4;
    end
    y = double(1/ductility);
end

function y = netFunctionStrainEnergy(net,x,strain)
%     x(1,dofsOfBoundaryNodes) = 0;
%     x(:) = normalize(x(:), 'center','mean');
    yPred = smooth(predict(net.net,x));
%     idx1 = find(yPred<=0.30*max(yPred));
%     idx2 = find(yPred==max(yPred));
%     idx3 = idx1(find(idx1>idx2,1));
%     yPred(idx3:end) = 0;
    if max(yPred) >=9.45 && max(yPred)<=9.975
        StrainEnergy = trapz(strain,yPred);
    else
        StrainEnergy = 1e-4;
    end
    y = double(1/StrainEnergy);
end
