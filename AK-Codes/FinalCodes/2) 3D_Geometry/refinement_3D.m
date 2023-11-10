function [nodesR,elementR] = refinement_3D(nodesI,element,division,L,H,W)


if (division > 1)
    elementI = [];
    numElem = size(element,1);
    for ii = 1:numElem
        elem = element(ii,:);
        for jj = 1:(division-1)
            if (jj==1)
                coord1 = nodesI(elem(2),2:4);
                coord2 = nodesI(elem(3),2:4);
                
                x3 = coord1(1) + (jj/division)*(coord2(1)-coord1(1));
                y3 = coord1(2) + (jj/division)*(coord2(2)-coord1(2));
                z3 = coord1(3) + (jj/division)*(coord2(3)-coord1(3));
                
                nodesI   = [nodesI;nodesI(end,1)+1 x3 y3 z3];
                elementI = [elementI;elem(2) nodesI(end,1)];
            elseif (jj==(division-1))
                coord1 = nodesI(elem(2),2:4);
                coord2 = nodesI(elem(3),2:4);
                
                x3 = coord1(1) + (jj/division)*(coord2(1)-coord1(1));
                y3 = coord1(2) + (jj/division)*(coord2(2)-coord1(2));
                z3 = coord1(3) + (jj/division)*(coord2(3)-coord1(3));
                
                nodesI   = [nodesI;nodesI(end,1)+1 x3 y3 z3];
                elementI = [elementI;nodesI(end,1)-1 nodesI(end,1)];
                elementI = [elementI;nodesI(end,1) elem(3)];
            else
                coord1 = nodesI(elem(2),2:4);
                coord2 = nodesI(elem(3),2:4);
                
                x3 = coord1(1) + (jj/division)*(coord2(1)-coord1(1));
                y3 = coord1(2) + (jj/division)*(coord2(2)-coord1(2));
                z3 = coord1(3) + (jj/division)*(coord2(3)-coord1(3));
                
                nodesI   = [nodesI;nodesI(end,1)+1 x3 y3 z3];
                elementI = [elementI;nodesI(end,1)-1 nodesI(end,1)];
            end
        end
    end
    elementR = [transpose(1:size(elementI,1)) elementI(:,1) elementI(:,2)];
    nodesR = nodesI;
else
    nodesR = nodesI;
    elementR = element;
end

% Find boundary nodes
xCoord = nodesR(:,2);
yCoord = nodesR(:,3);
zCoord = nodesR(:,4);
leftFace  = find(xCoord==0);
rightFace = find(xCoord==L);
bottomFace= find(yCoord==0);
topFace   = find(yCoord==H);
frontFace = find(zCoord==0);
backFace  = find(zCoord==W);

% AllSides = [leftFace rightFace bottomFace topFace frontFace backFace];

end