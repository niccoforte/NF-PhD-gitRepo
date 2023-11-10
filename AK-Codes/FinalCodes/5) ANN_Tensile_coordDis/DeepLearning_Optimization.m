clear all
close all
clc

networkDiagrams = 'D:\ArtificialNeuralNetwork\DistortedNodes\ANN';
meshFileLoc     = 'D:\MLCourse\LatticesDuctility\MATLAB_CNN\DeepLearningMATLAB\Network\FailureMech\StrainFieldPostProcessing';

startRow = 18;
endRow = 430;
perNodes            = import_input_file('FCC_12X16_per.inp',startRow,endRow);
perNodesCoordinates = table2array(perNodes(:,2:3));
perfectNodes = table2array(perNodes(:,1:3));
perfectNodes(:,1) = 1:1:endRow-startRow+1;
topNodes = find(perNodesCoordinates(:,2)==160);
botNodes = find(perNodesCoordinates(:,2)==0);
leftNodes = find(perNodesCoordinates(:,1)==0);
rightNodes = find(perNodesCoordinates(:,1)==120);

boundaryNodes = [topNodes;botNodes;leftNodes;rightNodes];
nonBoundaryNodes = find(~ismember(1:413,unique(boundaryNodes))==1);
dofsOfBoundaryNodes = [2*boundaryNodes-1;2*boundaryNodes]';

deltaXY1 = importdata(strcat(['inputData-',num2str(10),'-frame-',num2str(0),'.mat']));
deltaXY2 = importdata(strcat(['inputData-',num2str(20),'-frame-',num2str(0),'.mat']));
deltaXY3 = importdata(strcat(['inputData-',num2str(30),'-frame-',num2str(0),'.mat']));

deltaXY = [deltaXY2];
deltaXY(:,topNodes*2) = 0;

deltaXY(:,dofsOfBoundaryNodes) = [];

frames = [1:1:200];
len = size(deltaXY,2);
strain = linspace(0,0.01,201);
strain = strain(1,frames);

cd(networkDiagrams)
net = load('networkStressStrain.mat');


lbVal = -1*max(deltaXY(1000,:));
ubVal = +1*max(deltaXY(1000,:));

lb = ones(1,len)*lbVal;
% lb(1,2*boundaryNodes) = -Inf;
% lb(1,2*boundaryNodes-1) = -Inf;
ub = ones(1,len)*ubVal;
% ub(1,2*boundaryNodes) = Inf;
% ub(1,2*boundaryNodes-1) = Inf;


Aeq = zeros(len,len);
for kk = dofsOfBoundaryNodes
    Aeq(kk,kk) = 1;
end
beq(1,len) = 0;
A = [];
b = [];
nonlcon = [];

syms z [1 len]
x0 = deltaXY(1843,:);
% fun =@(z)netFunction(net,z,strain,dofsOfBoundaryNodes);
funSE =@(z)netFunctionStrainEnergy(net,z,strain,dofsOfBoundaryNodes);
% funMultiObj =@(z)netFunctionMultiObj(net,z,strain,dofsOfBoundaryNodes);

% Find minimum of function using simulated annealing algorithm
% rng default % For reproducibility
options = optimoptions('simulannealbnd','PlotFcns',...
          {@saplotbestx,@saplotbestf,@saplotx,@saplotf},'MaxIterations',10000);
% options.ObjectiveLimit  = 2.5;
% options.HybridFcn = 'fmincon';
[deltaOpt, fvalDuct] = simulannealbnd(funSE,x0,lb,ub,options);

deltaOptOriginal = deltaOpt;
deltaOpt = deltaOpt(1,:);
fvalDuct = fvalDuct(1,:);


% deltaOpt = (normalize(deltaOpt(:), 'center','mean'))';

yOptPred = predict(net.net,deltaOpt);
yPerPred = predict(net.net,x0);
SE_or_ductility = 1/fvalDuct(1)
% ductility2 = 1/fvalDuct(2)
% ductility3 = 1/fvalDuct(3)

figure
hold on; box on; grid on;
plot(strain,smooth(yPerPred),'LineWidth',1.5)
plot(strain,smooth(yOptPred),'LineWidth',1.5)
legend('Perfect True', 'Optimized Pred')
xlabel('Strain')
ylabel('Stress (MPa)')
xlim([0 0.01])
ylim([0 12])
set(gca,'FontSize',14)
set(gcf,'color','w')
hAx = gca;             
hAx.XAxis.Exponent=0;  


optNodes = zeros(size(perNodes));
optNodes(:,1) = 1:1:413;
optNodes(unique(boundaryNodes),2) = perNodesCoordinates(unique(boundaryNodes),1);
optNodes(unique(boundaryNodes),3) = perNodesCoordinates(unique(boundaryNodes),2);
optNodes(nonBoundaryNodes,2) = perNodesCoordinates(nonBoundaryNodes,1)+deltaOptOriginal(1:2:end)';
optNodes(nonBoundaryNodes,3) = perNodesCoordinates(nonBoundaryNodes,2)+deltaOptOriginal(2:2:end)';

figure
hold on
plot(perNodesCoordinates(:,1),perNodesCoordinates(:,2),'r*','LineWidth',1.5)
plot(optNodes(:,2)',optNodes(:,3)','bx','LineWidth',1.5)
legend('Perfect', 'Optimized')
set(gca,'FontSize',14)
set(gcf,'color','w')
axis off
axis equal

cd(meshFileLoc)
global node element
[nodes,nodesI] = node_new(120,160,12,16,0);
[connect] = triElemConnectivity(120,160,12,16,perfectNodes,nodes);
node = optNodes(:,2:3);
element = connect;
figure
hold on
axis off
fac = 10;
plot_mesh(node,element,'T3','r-');
cd(networkDiagrams)
% save('optimizedNodalCoordinates.mat','optNodes')

function y = netFunction(net,x,strain,dofsOfBoundaryNodes)
%     x(1,dofsOfBoundaryNodes) = 0;
%     x(:) = normalize(x(:), 'center','mean');
    yPred = smooth(predict(net.net,x));
    if max(yPred) >= 9.45 && max(yPred) <= 9.9750
        idx1 = find(yPred<=0.25*max(yPred));
        idx2 = find(yPred==max(yPred));
        ductility = strain(idx1(find(idx1>max(idx2),1)));
    else
        ductility = 1e-4;
    end
    y = double(1/ductility);
end

function y = netFunctionStrainEnergy(net,x,strain,dofsOfBoundaryNodes)
%     x(1,dofsOfBoundaryNodes) = 0;
%     x(:) = normalize(x(:), 'center','mean');
    yPred = smooth(predict(net.net,x));
%     idx1 = find(yPred<=0.30*max(yPred));
%     idx2 = find(yPred==max(yPred));
%     idx3 = idx1(find(idx1>idx2,1));
%     yPred(idx3:end) = 0;
    if max(yPred) >= 9.45 && max(yPred) <= 9.9750
        StrainEnergy = trapz(strain,yPred);
    else
        StrainEnergy = 1e-4;
    end
    y = double(1/StrainEnergy);
end