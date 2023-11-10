function [element,bElem] = connectivity_hex(L,H,unitCellSize,nnx,nny,nodes,TYPE,perError1)

radius = unitCellSize + 1e-3;
numElem = (nnx+1)*(nny+1);


dummyElem = [];
count = 0;
for ii = 1:length(nodes)
    distance = sqrt((nodes(ii,2)-nodes(:,2)).^2 +...
        (nodes(ii,3)-nodes(:,3)).^2);
    inside = find(distance<=radius);
    nearNodes = setdiff(inside,ii);
    for jj = 1:length(nearNodes)
        count = count + 1;
        dummyElem(count,1) = count;
        dummyElem(count,2:3) = [ii nearNodes(jj)];
    end
end


% Find boundary nodes
xCoord = nodes(:,2);
yCoord = nodes(:,3);
bottomNodes = find(yCoord>=-1e-3 & yCoord<=+1e-3);
topNodes    = find(yCoord>=H-1e-3 & yCoord<=H+1e-3);
leftNodes   = find(xCoord>=-1e-3 & xCoord<=+1e-3);
rightNodes  = find(xCoord>=L-1e-3 & xCoord<=L+1e-3);

% for ik = 1:(size(leftNodes,1)-1)
%     count = count + 1;
%     dummyElem(count,1) = count;
%     dummyElem(count,2:3) = [leftNodes(ik) leftNodes(ik+1)];
% end
% 
% for ik = 1:(size(rightNodes,1)-1)
%     count = count + 1;
%     dummyElem(count,1) = count;
%     dummyElem(count,2:3) = [rightNodes(ik) rightNodes(ik+1)];
% end

count = 0;
for i = 1:size(dummyElem,1)
    for j = 1:size(dummyElem,1)
        if (dummyElem(i,2)==dummyElem(j,3))
            if (dummyElem(i,3)==dummyElem(j,2))
                count = count + 1;
                dummyElem(j,:) = [0 0 0];
                break
            end
        end
    end
end

realElem1 = dummyElem;
indexRemove = [];
for i = 1:size(dummyElem,1)
    if (dummyElem(i,1)==0)
        indexRemove = [indexRemove i];
    end
end

realElem1(indexRemove,:) = [];
for i = 1:size(realElem1,1)
    realElem1(i,1) = i;
end
realElem2 = realElem1;



if (strcmpi(TYPE,'YES'))
    
    boundaryNodes     = unique([bottomNodes;topNodes;...
        leftNodes;rightNodes]);
    row1  = find((ismember(realElem2(:,2),boundaryNodes)==1));
    row2  = find((ismember(realElem2(:,3),boundaryNodes)==1));
    
    nonBoundaryElem = unique([row1;row2]);
    
    removeElem = realElem2;
    removeElem(nonBoundaryElem,:) = [];
    
    perError = int16(perError1*size(removeElem,1)*0.01) + 0.01*perError1*(6*nny + 4*nnx);
    missingElem = randi(size(removeElem,1),1,perError);
    
    realElem2(removeElem(missingElem,1),:) = [];
    for i = 1:size(realElem2,1)
        realElem2(i,1) = i;
    end
end


bElem = [];
bottomElemN1 = ismember(realElem2(:,2),bottomNodes)==1 & ismember(realElem2(:,3),bottomNodes)==1;
bottomElemN2 = find(bottomElemN1==1);
bElem = [bElem;realElem2(bottomElemN2,:)];

topElemN1 = ismember(realElem2(:,2),topNodes)==1 & ismember(realElem2(:,3),topNodes)==1;
topElemN2 = find(topElemN1==1);
bElem = [bElem;realElem2(topElemN2,:)];

leftElemN1 = ismember(realElem2(:,2),leftNodes)==1 & ismember(realElem2(:,3),leftNodes)==1;
leftElemN2 = find(leftElemN1==1);
bElem = [bElem;realElem2(leftElemN2,:)];

rightElemN1 = ismember(realElem2(:,2),rightNodes)==1 & ismember(realElem2(:,3),rightNodes)==1;
rightElemN2 = find(rightElemN1==1);
bElem = [bElem;realElem2(rightElemN2,:)];

element = realElem2;
end