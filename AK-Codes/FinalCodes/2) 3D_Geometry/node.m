function [nodes] = node(L,H,nnx,nny)

unitX = L/nnx;
unitY = H/nny;

totalNodes = (nnx+1)*(nny+1) + nnx*nny;
nodes = zeros(totalNodes,3);

count = 0;
x = 0; y = 0;
for i = 1:(nny+1)
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

x = unitX/2; y = unitY/2;
for i = 1:(nny)
    for j = 1:(nnx)
        count = count + 1;
        nodes(count,1) = count;
        nodes(count,2) = x;
        nodes(count,3) = y;
        x = x + unitX;
    end
    y = y + unitY;
    x = unitX/2;
end


end