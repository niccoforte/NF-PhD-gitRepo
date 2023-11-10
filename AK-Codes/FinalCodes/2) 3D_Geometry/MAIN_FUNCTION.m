clear all
close all
clc

unitType = 'fcc';
elementType = 'B21';

unitCellSize = 4;
nnx = 12;
nny = 16;
nnz = 1;
fac = 0.00;
division = 4;
L = unitCellSize*nnx;
H = unitCellSize*nny;
W = unitCellSize*nnz;

missingStruts = 'no';
perDefects     = 10;

SpringStiffness = 1e-5*69300*L;
maxDisp       = L*0.5;

sizeEffect_new = 'no';
solverType = 'standard';


[nodes,nodesI] = node_new_3D(L,H,W,nnx,nny,nnz,fac);

[element] = connectivity_FCC_3D(L,H,W,nnx,nny,nnz,nodes,nodesI,missingStruts,perDefects);


[nodes,element] = refinement_3D(nodesI,element,division,L,H,W);
% AllSoides = topNodes bottomNodes leftNodes rightNodes
[topNodes, bottomNodes, leftNodes, rightNodes, frontNodes, backNodes] = AllSidesCollect(nodes,L,H,W);

    
% AllSides = [topNodes, bottomNodes, leftNodes, rightNodes, frontNodes, backNodes];

FileName = 'Perfect_00.inp';
% TEXT_READ

nodeNum = nodes(:,1);
nodeX   = nodes(:,2);
nodeY   = nodes(:,3);
nodeZ   = nodes(:,4);

elemNum = element(:,1);
elem_1  = element(:,2);
elem_2  = element(:,3);

delete Nodes.txt
delete Elements.txt

% diary Nodes.txt
% for i = 1:length(nodeNum)
%     node_TEXT    = '%d, %f, %f, %f\n';
%     fprintf(node_TEXT,nodeNum(i),nodeX(i),nodeY(i),nodeZ(i));
% end
% diary off
% 
% diary Elements.txt
% for i = 1:length(elemNum)
%     node_TEXT    = '%d, %d, %d\n';
%     fprintf(node_TEXT,elemNum(i),elem_1(i),elem_2(i));
% end
% diary off
% 
% TEXT_READ
% 

% for i = 1:size(element,1)
%     coord1 = nodesI(element(i,2),2:4);
%     coord2 = nodesI(element(i,3),2:4);
%     figure(1000)
%     hold on
%     axis square
%     axis off
%     randThick = 0.1 + (1.9 - 0.1).*rand(1,1);
%     plot3([coord1(1);coord2(1)],[coord1(2);coord2(2)],[coord1(3);coord2(3)],'-b','LineWidth',2*randThick)
% end
% figure(1000)
% hold on
% axis equal
% axis off
% set(gcf,'color','w');
% view(61,16)

for i = 1:size(element,1)
    coord1 = nodes(element(i,2),2:3);
    coord2 = nodes(element(i,3),2:3);
    figure(2000)
    hold on
    axis square
    axis off
    plot([coord1(1);coord2(1)],[coord1(2);coord2(2)],'-b','LineWidth',2.0)
end
figure(2000)
hold on
axis square
axis off
% scatter3(nodesR(:,2),nodesR(:,3),nodesR(:,4),'r','.','LineWidth',2.0)
set(gcf,'color','w');
% view(61,16)

% tensile = transpose(15*cos((pi/180)*(0:15:90)));
% shear   = transpose(15*sin((pi/180)*(0:15:90)));