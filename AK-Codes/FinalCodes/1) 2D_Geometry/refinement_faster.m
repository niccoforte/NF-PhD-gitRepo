function [nodesR,elementR,thinElemArray] = refinement_faster(nodesI,element,division)


if (division > 1)
    elementI = zeros(length(element)*division,2);
    refineNodes = zeros((division-1)*length(element),2);
    numNodes = length(nodesI);
    thinElemArray=[];
    numElem = size(element,1);
    count = 0;
    for ii = 1:numElem
        elem = element(ii,:);
        for jj = 1:(division-1)
            if (jj==1)
                count = count + 1;
                coord1 = nodesI(elem(2),2:3);
                coord2 = nodesI(elem(3),2:3);
                
                x3 = coord1(1) + (jj/division)*(coord2(1)-coord1(1));
                y3 = coord1(2) + (jj/division)*(coord2(2)-coord1(2));
                
                refineNodes(count,:)  = [x3 y3];
                %elementI(count,:) = [elem(2) numNodes+count];
                
            elseif (jj==(division-1))
                count = count + 1;
                coord1 = nodesI(elem(2),2:3);
                coord2 = nodesI(elem(3),2:3);
                
                x3 = coord1(1) + (jj/division)*(coord2(1)-coord1(1));
                y3 = coord1(2) + (jj/division)*(coord2(2)-coord1(2));
                
                refineNodes(count,:)   = [x3 y3];
                %elementI(count,:) = [numNodes+count-1 elem(3)];
                %                 if (strcmpi(sizeEffect,'YES'))
                %                     if ismember(elem(1),thinElem)
                %                         thinElemArray = [thinElemArray;length(elementI)];
                %                     end
                %                 end
                %                 elementI = [elementI;nodesI(end,1) elem(3)];
                %                 if (strcmpi(sizeEffect,'YES'))
                %                     if ismember(elem(1),thinElem)
                %                         thinElemArray = [thinElemArray;length(elementI)];
                %                     end
                %                 end
            else
                count = count + 1;
                coord1 = nodesI(elem(2),2:3);
                coord2 = nodesI(elem(3),2:3);
                
                x3 = coord1(1) + (jj/division)*(coord2(1)-coord1(1));
                y3 = coord1(2) + (jj/division)*(coord2(2)-coord1(2));
                
                refineNodes(count,:)   = [x3 y3];
                %elementI(count,:) = [numNodes+count-1 numNodes+count];
                
            end
        end
    end
    count = 0;updateNode = numNodes; elementS = zeros(division,2);
    for ii = 1:numElem
        elem = element(ii,:);
        elementS(:,1) = [elem(2),(updateNode+1):1:(updateNode+division-1)];
        elementS(:,2) = [(updateNode+1):1:(updateNode+division-1),elem(3)];
        elementI((division*count+1):(ii*division),:) = elementS;
        updateNode = updateNode + division - 1;
        count = count + 1;
%         for jj = 1:(division)
%             if (jj==1)
%                 count = count + 1;
%                 elementI(count,:) = [elem(2) numNodes+count];
%             elseif (jj==(division))
%                 count = count + 1;
%                 elementI(count,:) = [numNodes+count-1 elem(3)];
%             else
%                 count = count + 1;
%                 elementI(count,:) = [numNodes+count-1 numNodes+count];
%             end
%         end
    end
    elementR = [transpose(1:size(elementI,1)) elementI(:,1) elementI(:,2)];
    refineNodes = [transpose(1:size(refineNodes,1))+length(nodesI) refineNodes(:,1) refineNodes(:,2)];
    nodesI = [nodesI;refineNodes];
    nodesR = nodesI;
else
    nodesR = nodesI;
    elementR = element;
    thinElemArray = [];
end

end