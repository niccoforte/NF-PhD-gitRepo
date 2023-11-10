function [connect] = SolidTriElem(L,H,unitCellSize,nnx,nny,nodes)

unitX = unitCellSize;
unitY = sqrt(3)*unitCellSize;

numFreeHexNodes = nnx*nny + floor(nny/2);
hexNodes = zeros(numFreeHexNodes,3);

count = 0; y = unitY/2;
for i = 1:(nny)
    if mod(i,2)==1
        x = 0.5*unitX;
        for j = 1:(nnx)
            count = count + 1;
            hexNodes(count,1) = count;
            hexNodes(count,2) = x;
            hexNodes(count,3) = y;
            x = x + 2*unitX;
        end
        y = y + unitY;
    else
        x = -0.5*unitX;
        for j = 1:(nnx+1)
            count = count + 1;
            hexNodes(count,1) = count;
            hexNodes(count,2) = x;
            hexNodes(count,3) = y;
            x = x + 2*unitX;
        end
        y = y + unitY;
    end
end

radius = unitX + 1e-3;
connectHex = zeros(numFreeHexNodes,3);
count = 0;
for ii = 1:length(hexNodes)
    distance = sqrt((hexNodes(ii,2)-nodes(:,2)).^2 +...
        (hexNodes(ii,3)-nodes(:,3)).^2);
    inside = find(distance<=radius);
    
    if length(inside)==3
        count = count + 1;
        connectHex(count,:) = sort(inside)';
    elseif length(inside)==5
        if nodes(inside(5),2) < L/2
            count = count + 1;
            connectHex(count,:) = [inside(1) inside(2) inside(4)];
            count = count + 1;
            connectHex(count,:) = [inside(1) inside(4) inside(3)];
            count = count + 1;
            connectHex(count,:) = [inside(2) inside(5) inside(4)];
        elseif nodes(inside(5),2) > L/2
            count = count + 1;
            connectHex(count,:) = [inside(1) inside(2) inside(4)];
            count = count + 1;
            connectHex(count,:) = [inside(1) inside(4) inside(3)];
            count = count + 1;
            connectHex(count,:) = [inside(1) inside(3) inside(5)];
        else
            disp('Error in finding node location for triangulization')
        end
    elseif length(inside)==6
        count = count + 1;
        connectHex(count,:) = [inside(1) inside(2) inside(6)];
        count = count + 1;
        connectHex(count,:) = [inside(6) inside(4) inside(3)];
        count = count + 1;
        connectHex(count,:) = [inside(1) inside(3) inside(5)];
        count = count + 1;
        connectHex(count,:) = [inside(1) inside(6) inside(3)];
    else
        disp('Error in number of nodes for triangulization')
    end       
end

numFreeTriNodes = 2*((nnx)*floor(nny/2) + (nnx-1)*round(nny/2));
triNodes = zeros(numFreeTriNodes,3);

count = 0; y = unitY/2;
for i = 1:(nny)
    if mod(i,2)==1
        x = 1.5*unitX;
        for j = 1:(nnx-1)
            count = count + 1;
            triNodes(count,1) = count;
            triNodes(count,2) = x;
            triNodes(count,3) = y - (sqrt(3)/3)*unitX;
            count = count + 1;
            triNodes(count,1) = count;
            triNodes(count,2) = x;
            triNodes(count,3) = y + (sqrt(3)/3)*unitX;
            x = x + 2*unitX;
        end
        y = y + unitY;
    else
        x = 0.5*unitX;
        for j = 1:(nnx)
            count = count + 1;
            triNodes(count,1) = count;
            triNodes(count,2) = x;
            triNodes(count,3) = y - (sqrt(3)/3)*unitX;
            count = count + 1;
            triNodes(count,1) = count;
            triNodes(count,2) = x;
            triNodes(count,3) = y + (sqrt(3)/3)*unitX;
            x = x + 2*unitX;
        end
        y = y + unitY;
    end
end

radius = (sqrt(3)/3)*unitX + 1e-3;
connectTri = zeros(numFreeTriNodes,3);
for ii = 1:length(triNodes)
    distance = sqrt((triNodes(ii,2)-nodes(:,2)).^2 +...
        (triNodes(ii,3)-nodes(:,3)).^2);
    inside = find(distance<=radius);
    connectTri(ii,:) = sort(inside)';
end

connect = [connectHex; connectTri];
