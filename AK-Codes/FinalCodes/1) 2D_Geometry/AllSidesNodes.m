function [topNodes, bottomNodes, leftNodes, rightNodes] = AllSidesNodes(nodes,L,H)
% Find boundary nodes
xCoord = nodes(:,2);
yCoord = nodes(:,3);
bottomNodes = find(yCoord>=-1e-3 & yCoord<=+1e-3);
topNodes    = find(yCoord>=H-1e-3 & yCoord<=H+1e-3);
leftNodes   = find(xCoord>=-1e-3 & xCoord<=+1e-3);
rightNodes  = find(xCoord>=L-1e-3 & xCoord<=L+1e-3);
end
