clear all
close all
clc

unitType = 'FCC';
elementType = 'B21';

density = 2.7e-9;
RD = 0.25;
nnx = 60;

if (strcmpi(unitType,'kagome'))
    unitCellSize = 10;
    nnx = nnx+1;
    nny = nnx;
    L = unitCellSize*(2*nnx - 1);
    H = sqrt(3)*unitCellSize*nny;
elseif (strcmpi(unitType,'FCC'))
    unitCellSize = 10;
    nnx = 10;
    nny = 10;
    L = unitCellSize*nnx;
    H = unitCellSize*nny;
    crossSec = 'rect';
elseif (strcmpi(unitType,'Diamond'))
    unitCellSize = 20;
    nnx = nnx;
    nny = nnx;
    L = unitCellSize*nnx;
    H = unitCellSize*nny;
elseif (strcmpi(unitType,'Hexagonal'))
    unitCellSize = 10;
    nnx = nnx+1;
    nny = 15;
    L = sqrt(3)*unitCellSize*nnx;
    H = 2*unitCellSize*round(nny/2)+unitCellSize*floor(nny/2);
end

fac = 0.0;
division = 10;
maxDisp       = L*0.25;

% Sxx = [1 0.5 0.001 -0.5 -1 -1.0 -1.000 -1.0 -1 0.2 0.1];
% Syy = [1 1.0 1.000 +1.0 +1 +0.5 +0.001 -0.5 -1 1.0 1.0];

Sxx = [0.0001 -0.2679 -0.5774 -1];
Syy = [1.0000 +1.0000 +1.0000 +1];
rho = Sxx./Syy;

SpringStiffness = 1e-7*71300*L*100;
Amplitude = nnx/60;

missingStruts = 'yes';
sizeEffect_new = 'no';
solverType = 'explicit';

perDefects     = 2;



volumeList = zeros(20,2); stiffness = zeros(20,6);
for iteration = 1
    
    if (strcmpi(unitType,'kagome'))
        [nodes,nodesI] = node_kagome(L,H,unitCellSize,nnx,nny,fac);
        [element,bElem] = connectivity_kagome(L,H,unitCellSize,nnx,nny,nodes,missingStruts,perDefects);
        
%         element = []; count = 0;
%         for del = 1:length(elementOld)
%             if ~ismember(del,bElem(:,1))
%                 count = count + 1;
%                 element(count,:) = [count elementOld(del,2:3)];
%             end
%         end
        
        len = zeros(length(element),1);
        for ik = 1:length(element)
            x1 = nodesI(element(ik,2),2);
            x2 = nodesI(element(ik,3),2);
            y1 = nodesI(element(ik,2),3);
            y2 = nodesI(element(ik,3),3);
            len(ik,1) = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
        end
        
        rad = 2*RD*L*H/(sum(len)*pi);
        rveVol = (L)*(H)*rad*2;
        
        [N] = nMatrix_new(element,nodesI(:,2:3));
        c_not = pi*rad*rad*(diag(len))./rveVol;
        A = transpose(c_not*N)*N; C = inv(A);
        
        [connect] = SolidTriElem(L,H,unitCellSize,nnx,nny,nodes);
        
%         element = elementI;
        location = 'E:\ShearBandAllLattice\ABAQUS\Kagome'; 
        FileName = strcat([unitType,'_',num2str(nnx-1),'X',num2str(nnx-1),'_',...
            num2str(perDefects),'.inp']);
    
    elseif (strcmpi(unitType,'FCC'))
        [nodes,nodesI] = node_new(L,H,nnx,nny,fac);
        [element] = connectivity_FCC(L,H,nnx,nny,nodes,missingStruts,perDefects,unitType);
        
        len = zeros(length(element),1);
        for ik = 1:length(element)
            x1 = nodesI(element(ik,2),2);
            x2 = nodesI(element(ik,3),2);
            y1 = nodesI(element(ik,2),3);
            y2 = nodesI(element(ik,3),3);
            len(ik,1) = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
        end
        if (strcmpi(crossSec,'circ'))
            p = [4*RD (L+H-pi*sum(len)) 4*RD*(L*H)];
            dia_opt = roots(p);
            dia_est = 2*RD*4*L*H/(sum(len)*2*pi);
            diff_sqr = [(dia_opt(1)-dia_est)^2 (dia_opt(2)-dia_est)^2]; [minVal,index] = min(diff_sqr);
            rad = dia_opt(index)./2;
            rveVol = (L+2*rad)*(H+2*rad)*rad*2;
        elseif (strcmpi(crossSec,'rect'))
            th = RD*L*H/sum(len);
            rad = th;
            rad1 = 1.2;
            rad2 = rad;
            rveVol = (L+2*rad)*(H+2*rad)*rad*2;
        end

        [N] = nMatrix_new(element,nodesI(:,2:3));
        c_not = pi*rad*rad*(diag(len))./rveVol;
        A = transpose(c_not*N)*N; C = inv(A);
        
%         location = 'E:\ShearBandAllLattice\ABAQUS\FCC';
        location = 'E:\ShearBandAllLattice\ABAQUS\RigTest';
%         FileName = strcat([unitType,'_',num2str(nnx),'X',num2str(nnx),'_',num2str(perDefects),'.inp']);
        
    elseif (strcmpi(unitType,'Diamond'))
        [nodes,nodesI] = node_new(L,H,nnx,nny,fac);
        [element] = connectivity_FCC(L,H,nnx,nny,nodes,missingStruts,perDefects,unitType);
        
        len = zeros(length(element),1);
        for ik = 1:length(element)
            x1 = nodesI(element(ik,2),2);
            x2 = nodesI(element(ik,3),2);
            y1 = nodesI(element(ik,2),3);
            y2 = nodesI(element(ik,3),3);
            len(ik,1) = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
        end
        
        rad = 2*RD*L*H/(sum(len)*pi);
        rveVol = (L)*(H)*rad*2;
        
        [N] = nMatrix_new(element,nodesI(:,2:3));
        c_not = pi*rad*rad*(diag(len))./rveVol;
        A = transpose(c_not*N)*N; C = inv(A);
        
        location = 'E:\ShearBandAllLattice\ABAQUS\Diamond';
        FileName = strcat([unitType,'_',num2str(nnx),'X',num2str(nnx),'_',num2str(perDefects),'.inp']);
        
    elseif (strcmpi(unitType,'Hexagonal'))
        [nodes,nodesI] = node_hex(L,H,unitCellSize,nnx,nny,fac);
        [element,bElem] = connectivity_hex(L,H,unitCellSize,nnx,nny,nodes,missingStruts,perDefects);
        
        len = zeros(length(element),1);
        for ik = 1:length(element)
            x1 = nodesI(element(ik,2),2);
            x2 = nodesI(element(ik,3),2);
            y1 = nodesI(element(ik,2),3);
            y2 = nodesI(element(ik,3),3);
            len(ik,1) = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
        end
        
        rad = 2*RD*L*H/(sum(len)*pi);
        rveVol = (L)*(H)*rad*2;
        
        [N] = nMatrix_new(element,nodesI(:,2:3));
        c_not = pi*rad*rad*(diag(len))./rveVol;
        A = transpose(c_not*N)*N; C = inv(A);
        
        location = 'E:\ShearBandAllLattice\ABAQUS\Hexagonal';
        FileName = strcat([unitType,'_',num2str(nnx-1),'X',num2str(nnx-1),'_',num2str(perDefects),'.inp']);
    end
    
    
    
    
    [nodes,element,thinElemArray] = refinement_faster(nodesI,element,division);
    
    [topNodes, bottomNodes, leftNodes, rightNodes] = AllSidesNodes(nodes,L,H);
    
    %     cn1 = intersect(bottomNodes,leftNodes);
    %     cn2 = intersect(bottomNodes,rightNodes);
    %     cn3 = intersect(rightNodes,topNodes);
    %     cn4 = intersect(topNodes,leftNodes);
    %
    %     nodes([cn1 cn2 cn3 cn4],:) = [];
    %     [topNodes, bottomNodes, leftNodes, rightNodes] = AllSidesNodes(nodes,L,H);
    
    % nodes(:,1) = nodes(1:end,1);
    
    
    for value = 1:length(rho)
        FileName = strcat([unitType,'_',num2str(nnx),'X',num2str(nnx),'_',num2str(value),'_missing.inp']);
        if (rho(value)>=0)
            if (Syy(value)>=0)
                loadingType = 'ten'; % tension = ten;
            else
                loadingType = 'bicom'; % biaxial compression = bicom;
            end
        else
            loadingType = 'com'; % compression = com;
        end
        
        if (strcmpi(solverType,'explicit'))
            TEXT_READ_V4
        elseif (strcmpi(solverType,'standard'))
            TEXT_READ_V3
        end
        
        movefile(FileName,(location))
    end
    
end

% for i = 1:size(element,1)
%     coord1 = nodes(element(i,2),2:3);
%     coord2 = nodes(element(i,3),2:3);
%     figure(1000)
%     hold on
%     if ismember(i,bElem(:,1))
%         plot([coord1(1);coord2(1)],[coord1(2);coord2(2)],'-k','LineWidth',2.0)
%     else
%         plot([coord1(1);coord2(1)],[coord1(2);coord2(2)],'-b','LineWidth',2.5)
%     end
% end
% figure(1000)
% hold on
% % axis square
% axis off
% set(gcf,'color','w');
% 
% elemType = 'T3';
% figure
% hold on
% plot_mesh(nodes(:,2:3),connect,elemType,'r-');
% axis equal
% axis off


% for i = 1:size(elementI,1)
%     coord1 = nodes(elementI(i,2),2:3);
%     coord2 = nodes(elementI(i,3),2:3);
%     figure(2000)
%     hold on
%     axis square
%     axis off
% %     if ismember(i,bElem(:,1))
% %         plot([coord1(1);coord2(1)],[coord1(2);coord2(2)],'-r','LineWidth',2.0)
% %     else
%         plot([coord1(1);coord2(1)],[coord1(2);coord2(2)],'-b','LineWidth',2.5)
% %     end
% end
% figure(2000)
% hold on
% % axis square
% axis off
% set(gcf,'color','w');

figure(3000)
hold on
axis square
axis off
plot(nodes(:,2),nodes(:,3),'.k')
plot(nodes(topNodes,2),nodes(topNodes,3),'*b')
plot(nodes(bottomNodes,2),nodes(bottomNodes,3),'*b')
plot(nodes(leftNodes,2),nodes(leftNodes,3),'*r')
plot(nodes(rightNodes,2),nodes(rightNodes,3),'*r')
%text(nodesR(:,2),nodesR(:,3),num2str(nodesR(:,1)))
set(gcf,'color','w');