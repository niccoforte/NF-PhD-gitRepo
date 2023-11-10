function [element] = connectivity(L,H,nnx,nny,nodes,fac)

radius = L/nnx;
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

% for removing horizontal connections
count = 0;
for i = 1:size(realElem1,1)
    tangent = (nodes(realElem1(i,3),3)-nodes(realElem1(i,2),3))/...
        (nodes(realElem1(i,3),2)-nodes(realElem1(i,2),2));
    if (tangent==0)
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


for i = 1:size(realElem2,1)
    coord1 = nodes(realElem2(i,2),2:3);
    coord2 = nodes(realElem2(i,3),2:3);
    figure(1)
    hold on
    axis square
    axis off
    plot([coord1(1);coord2(1)],[coord1(2);coord2(2)],'-b','LineWidth',2.0)
end
element = realElem2;
end