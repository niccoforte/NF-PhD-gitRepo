function [element] = connectivity_FCC(L,H,nnx,nny,nodes,TYPE,perError1,unitType)

radius = min(L,H)/min(nnx,nny);
numElem = (nnx+1)*(nny+1);


dummyElem = [];
count = 0;
for ii = 1:numElem
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

if (strcmpi(unitType,'Diamond'))
    % for removing horizontal connections
    count = 0;
    for i = 1:size(realElem1,1)
        tangent = (nodes(realElem1(i,3),3)-nodes(realElem1(i,2),3))/...
            (nodes(realElem1(i,3),2)-nodes(realElem1(i,2),2));
        if (tangent==0 || tangent==inf)
            realElem1(i,:) = [0 0 0];
        end
    end
    
    realElem2 = realElem1;
    indexRemove = [];
    for i = 1:size(realElem1,1)
        if (realElem1(i,1)==0)
            indexRemove = [indexRemove i];
        end
    end
    realElem2(indexRemove,:) = [];
    
    for i = 1:size(realElem2,1)
        realElem2(i,1) = i;
    end
    
elseif (strcmpi(unitType,'Grid'))
    % for removing horizontal connections
    count = 0;
    for i = 1:size(realElem1,1)
        tangent = (nodes(realElem1(i,3),3)-nodes(realElem1(i,2),3))/...
            (nodes(realElem1(i,3),2)-nodes(realElem1(i,2),2));
        if ~(tangent==0 || tangent==inf)
            realElem1(i,:) = [0 0 0];
        end
    end
    
    realElem2 = realElem1;
    indexRemove = [];
    for i = 1:size(realElem1,1)
        if (realElem1(i,1)==0)
            indexRemove = [indexRemove i];
        end
    end
    realElem2(indexRemove,:) = [];
    
    for i = 1:size(realElem2,1)
        realElem2(i,1) = i;
    end
else
    realElem2 = realElem1;
end



if (strcmpi(TYPE,'YES'))
    
    % Find boundary nodes
    xCoord = nodes(:,2);
    yCoord = nodes(:,3);
    bottomNodes = find(yCoord==0);
    topNodes    = find(yCoord==H);
    leftNodes   = find(xCoord==0);
    rightNodes  = find(xCoord==L);
    
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

element = realElem2;
end