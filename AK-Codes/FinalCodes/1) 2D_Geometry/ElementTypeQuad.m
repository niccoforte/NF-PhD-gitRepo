function [nodesR,elementR] = ElementTypeQuad(nodes,element)
midNodesNum = [max(nodes(:,1))+1:max(nodes(:,1))+size(element,1)]';
nodesR = zeros(size(nodes,1)+size(element,1),3);
nodesR(1:size(nodes,1),1) = nodes(:,1);
nodesR(1:size(nodes,1),2) = nodes(:,2);
nodesR(1:size(nodes,1),3) = nodes(:,3);
nodesR(midNodesNum,1) = midNodesNum;
nodesR(midNodesNum,2) = 0.5*(nodes(element(:,2),2) + nodes(element(:,3),2));
nodesR(midNodesNum,3) = 0.5*(nodes(element(:,2),3) + nodes(element(:,3),3));

elementR = zeros(size(element,1),4);
elementR(:,1) = element(:,1);
elementR(:,2) = element(:,2);
elementR(:,3) = midNodesNum;
elementR(:,4) = element(:,3);
end