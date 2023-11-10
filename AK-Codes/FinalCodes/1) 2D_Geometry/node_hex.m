function [nodes,nodesR] = node_hex(L,H,unitCellSize,nnx,nny,fac)

unitX = sqrt(3)*unitCellSize;
unitY = 2*unitCellSize;

delta = unitCellSize*fac;

totalNodes = 2*(nnx)*round(nny/2) + 2*(nnx+1)*round(nny/2);
nodes = zeros(totalNodes,3);

count = 0;
x = 0.5*sqrt(3)*unitCellSize; y = 0;
for i = 1:round(nny/2)
    for j = 1:(nnx)
        count = count + 1;
        nodes(count,1) = count;
        nodes(count,2) = x;
        nodes(count,3) = y;
        x = x + unitX;
    end
    y = y + unitY;
    x = 0.5*sqrt(3)*unitCellSize;
    for j = 1:(nnx)
        count = count + 1;
        nodes(count,1) = count;
        nodes(count,2) = x;
        nodes(count,3) = y;
        x = x + unitX;
    end
    y = y + unitCellSize;
    x = 0.5*sqrt(3)*unitCellSize;
end

x = 0; y = unitCellSize/2;
for i = 1:round(nny/2)
    for j = 1:(nnx+1)
        count = count + 1;
        nodes(count,1) = count;
        nodes(count,2) = x;
        nodes(count,3) = y;
        x = x + unitX;
    end
    y = y + unitCellSize;
    x = 0;
    for j = 1:(nnx+1)
        count = count + 1;
        nodes(count,1) = count;
        nodes(count,2) = x;
        nodes(count,3) = y;
        x = x + unitX;
    end
    y = y + unitY;
    x = 0;
end

% Find boundary nodes
xCoord = nodes(:,2);
yCoord = nodes(:,3);
bottomNodes = find(yCoord>=-1e-3 & yCoord<=+1e-3);
topNodes    = find(yCoord>=H-1e-3 & yCoord<=H+1e-3);
leftNodes   = find(xCoord>=-1e-3 & xCoord<=+1e-3);
rightNodes  = find(xCoord>=L-1e-3 & xCoord<=L+1e-3);

boundaryNodes     = unique([bottomNodes;topNodes;...
    leftNodes;rightNodes]);

nodeIndex = nodes(:,1);
nonboundaryNodes = setdiff(nodeIndex,boundaryNodes);

boundaryCoord    = nodes(boundaryNodes,2:3);
nonboundaryCoord = nodes(nonboundaryNodes,2:3);

% figure
% hold on
% axis square
% axis off
% scatter(boundaryCoord(:,1),boundaryCoord(:,2),100,'x','LineWidth',2.0)
% scatter(nonboundaryCoord(:,1),nonboundaryCoord(:,2),'r','.')

% Add random delta to the nonboundary coordinates

randX = -delta + (delta + delta).*rand(length(nonboundaryNodes),1);
randY = -delta + (delta + delta).*rand(length(nonboundaryNodes),1);

nonboundaryCoordX = nonboundaryCoord(:,1) + randX;
nonboundaryCoordY = nonboundaryCoord(:,2) + randY;

nonboundaryCoordNew = [nonboundaryCoordX nonboundaryCoordY];

% figure
% hold on
% axis square
% axis off
% scatter(boundaryCoord(:,1),boundaryCoord(:,2),100,'x','LineWidth',2.0)
% scatter(nonboundaryCoord(:,1),nonboundaryCoord(:,2),'r','.')
% scatter(nonboundaryCoordNew(:,1),nonboundaryCoordNew(:,2),'g','.')

nodesR = nodes;
nodesR(nonboundaryNodes,2:3) = [nonboundaryCoordX nonboundaryCoordY];

end
