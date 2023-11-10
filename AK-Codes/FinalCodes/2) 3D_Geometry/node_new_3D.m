function [nodes,nodesR] = node_new_3D(L,H,W,nnx,nny,nnz,fac)

unitX = L/nnx;
unitY = H/nny;
unitZ = W/nnz;

delta = 0.5*sqrt(unitX*unitX + unitY*unitY + unitZ*unitZ)*fac;

totalNodes = (nnx+1)*(nny+1)*(nnz+1) + nnx*nny*nnz;
nodes = zeros(totalNodes,4);

count = 0;
x = 0; y = 0;z = 0;
for i = 1:(nnz+1)
    for j = 1:(nny+1)
        for k = 1:(nnx+1)
            count = count + 1;
            nodes(count,1) = count;
            nodes(count,2) = x;
            nodes(count,3) = y;
            nodes(count,4) = z;
            x = x + unitX;
        end
        y = y + unitY;
        x = 0;
    end
    z = z + unitZ;
    x = 0;y = 0;
end

x = unitX/2; y = unitY/2; z = unitZ/2;

for k = 1:(nnz)
    for i = 1:(nny)
        for j = 1:(nnx)
            count = count + 1;
            nodes(count,1) = count;
            nodes(count,2) = x;
            nodes(count,3) = y;
            nodes(count,4) = z;
            x = x + unitX;
        end
        y = y + unitY;
        x = unitX/2;
    end
    z = z + unitZ;
    x = unitX/2; y = unitY/2;
end

% Find boundary nodes
xCoord = nodes(:,2);
yCoord = nodes(:,3);
zCoord = nodes(:,4);
leftFace  = find(xCoord==0);
rightFace = find(xCoord==L);
bottomFace= find(yCoord==0);
topFace   = find(yCoord==H);
frontFace = find(zCoord==W);
backFace  = find(zCoord==0);

boundaryNodes     = unique([leftFace;rightFace;...
    bottomFace;topFace;frontFace;backFace]);
% boundaryNodes     = unique([frontFace;backFace]);

nodeIndex = nodes(:,1);
nonboundaryNodes = setdiff(nodeIndex,boundaryNodes);

boundaryCoord    = nodes(boundaryNodes,2:4);
nonboundaryCoord = nodes(nonboundaryNodes,2:4);

figure
hold on
axis on
axis square
% scatter3(boundaryCoord(:,1),boundaryCoord(:,2),boundaryCoord(:,3),100,'x','LineWidth',2.0)
scatter3(nonboundaryCoord(:,1),nonboundaryCoord(:,2),nonboundaryCoord(:,3),'r','.')
scatter3(nodes(backFace,2),nodes(backFace,3),nodes(backFace,4),100,'x','LineWidth',2.0);
scatter3(nodes(frontFace,2),nodes(frontFace,3),nodes(frontFace,4),100,'^','LineWidth',2.0);
xlabel('x label')
ylabel('y label')
zlabel('z label')
view(40,15)

% Add random delta to the nonboundary coordinates

randX = -delta + (delta + delta).*rand(length(nonboundaryNodes),1);
randY = -delta + (delta + delta).*rand(length(nonboundaryNodes),1);
randZ = -delta + (delta + delta).*rand(length(nonboundaryNodes),1);

nonboundaryCoordX = nonboundaryCoord(:,1) + randX;
nonboundaryCoordY = nonboundaryCoord(:,2) + randY;
nonboundaryCoordZ = nonboundaryCoord(:,3) + randZ;

nonboundaryCoordNew = [nonboundaryCoordX nonboundaryCoordY nonboundaryCoordZ];

figure
hold on
axis square
axis on
scatter3(boundaryCoord(:,1),boundaryCoord(:,2),boundaryCoord(:,3),100,'x','LineWidth',2.0)
scatter3(nonboundaryCoord(:,1),nonboundaryCoord(:,2),nonboundaryCoord(:,3),'r','.')
scatter3(nonboundaryCoordNew(:,1),nonboundaryCoordNew(:,2),nonboundaryCoordNew(:,3),'g','.')
view(40,15)

nodesR = nodes;
nodesR(nonboundaryNodes,2:4) = [nonboundaryCoordX nonboundaryCoordY nonboundaryCoordZ];

end
