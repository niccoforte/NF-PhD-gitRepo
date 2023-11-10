function [nodes,nodesR] = node_kagome(L,H,unitCellSize,nnx,nny,fac)

unitX = unitCellSize;
unitY = sqrt(3)*unitCellSize;

delta = unitCellSize*fac;

totalNodes = (2*nnx*(nny+1)) + (nnx-1)*round(nny/2) + (nnx)*floor(nny/2);
nodes = zeros(totalNodes,3);

count = 0;
x = 0; y = 0;
for i = 1:(nny+1)
    for j = 1:(2*nnx)
        count = count + 1;
        nodes(count,1) = count;
        nodes(count,2) = x;
        nodes(count,3) = y;
        x = x + unitX;
    end
    y = y + unitY;
    x = 0;
end

x = 1.5*unitX; y = unitY/2;
for i = 1:(round(nny/2))
    for j = 1:(nnx-1)
        count = count + 1;
        nodes(count,1) = count;
        nodes(count,2) = x;
        nodes(count,3) = y;
        x = x + 2*unitX;
    end
    y = y + 2*unitY;
    x = 1.5*unitX;
end

x = 0.5*unitX; y = 1.5*unitY;
for i = 1:(floor(nny/2))
    for j = 1:(nnx)
        count = count + 1;
        nodes(count,1) = count;
        nodes(count,2) = x;
        nodes(count,3) = y;
        x = x + 2*unitX;
    end
    y = y + 2*unitY;
    x = unitX/2;
end

% Find boundary nodes
xCoord = nodes(:,2);
yCoord = nodes(:,3);
bottomNodes = find(yCoord==0);
topNodes    = find(yCoord>=H-1e-3 & yCoord<=H+1e-3);
leftNodes   = find(xCoord==0);
rightNodes  = find(xCoord==L);

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
