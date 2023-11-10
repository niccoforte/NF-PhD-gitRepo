function [topNodes, bottomNodes, leftNodes, rightNodes, frontNodes, backNodes] = AllSidesCollect(nodes,L,H,W)
% Find boundary nodes
xCoord = nodes(:,2);
yCoord = nodes(:,3);
zCoord = nodes(:,4);
backNodes = find(zCoord==0);
frontNodes= find(zCoord==W);
bottomNodes = find(yCoord==0);
topNodes    = find(yCoord==H);
leftNodes   = find(xCoord==0);
rightNodes  = find(xCoord==L);
end
