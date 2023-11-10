switch solverType
    case 'standard'
        if (strcmpi(sizeEffect_new,'YES'))
            if (strcmpi(missingStruts,'YES'))
                fid=fopen('Tension_Std_v3_150%D.inp');
            else
                fid=fopen('Tension_Std_v2_150%D.inp');
            end
        else
            if (strcmpi(missingStruts,'YES'))
                fid=fopen('Tension_Std_v3.inp');
            else
                fid=fopen('Tension_Std_v2.inp');
            end
        end
    case 'implicit'
        fid=fopen('Tension_Implicit.inp');
    case 'explicit'
        fid=fopen('Tension_Explicit.inp');
end
tline = fgetl(fid);
tlines = cell(0,1);
while ischar(tline)
    tlines{end+1,1} = tline;
    tline = fgetl(fid);
end
fclose(fid);

ID = strfind(tlines,'*Node');
indexNode = find(not(cellfun('isempty',ID)));
ID = strfind(tlines,'** Section');
indexSection = find(not(cellfun('isempty',ID)));
ID = strfind(tlines,'*End Assembly');
indexEndAssembly = find(not(cellfun('isempty',ID)));
ID = strfind(tlines,'** BOUNDARY CONDITIONS');
indexBoundary = find(not(cellfun('isempty',ID)));
ID = strfind(tlines,'** OUTPUT REQUESTS');
indexOutPutRequest = find(not(cellfun('isempty',ID)));
ID = strfind(tlines,'*End Step');
indexEnd = find(not(cellfun('isempty',ID)));

diary(FileName)
for i = 1:indexNode(1)
    TEXT = tlines(i);
    fprintf(char(TEXT));
    fprintf('\n');
end
for i = 1:length(nodes)
    node_TEXT    = '%d, %f, %f, %f\n';
    fprintf(node_TEXT,nodes(i,1),nodes(i,2),nodes(i,3),nodes(i,4));
end
% for i = 1:length(AllSides)
%     %fprintf('*Node\n');
%     node_TEXT    = '%d, %d, %d\n';
%     fprintf(node_TEXT,500000+(2*i-1),nodesR(AllSides(i,1),2), unitCellSize*nnx+50);
%     fprintf(node_TEXT,500000+2*i    ,unitCellSize*nny+50           ,nodesR(AllSides(i,4),3));
% end

if (strcmpi(elementType,'B21'))
    fprintf('\n');
    fprintf('*Element, type=B21\n')
    for i = 1:length(element)
        node_TEXT    = '%d, %d, %d\n';
        fprintf(node_TEXT,element(i,1),element(i,2),element(i,3));
    end
elseif (strcmpi(elementType,'B22')) % only works when no refinement needed
    fprintf('\n');
    fprintf('*Element, type=B22\n')
    for i = 1:length(element)
        node_TEXT    = '%d, %d, %d, %d\n';
        fprintf(node_TEXT,element(i,1),element(i,2),element(i,3),element(i,4));
    end
end


fprintf('*Elset, elset=SET-2, generate\n');
fprintf('1, ');
fprintf(num2str(length(element)));
fprintf(', 1\n');


for i = indexSection(1):indexNode(2)-1
    TEXT = tlines(i);
    fprintf(char(TEXT));
    fprintf('\n');
end

% pre_top = 't';
% names_top = {};
% 
% for k = 1:length(AllSides)
%     names_top = [names_top;strcat([pre_top,num2str(k,'%02d')])];
% end
% 
% pre_bot = 'b';
% names_bot = {};
% 
% for k = 1:length(AllSides)
%     names_bot = [names_bot;strcat([pre_bot,num2str(k,'%02d')])];
% end
% 
% pre_right = 'r';
% names_right = {};
% 
% for k = 1:length(AllSides)
%     names_right = [names_right;strcat([pre_right,num2str(k,'%02d')])];
% end
% 
% pre_left = 'l';
% names_left = {};
% 
% for k = 1:length(AllSides)
%     names_left = [names_left;strcat([pre_left,num2str(k,'%02d')])];
% end
% 
% pre_referenceTop = 'RT';
% names_referenceTop = {};
% 
% for k = 1:length(AllSides)
%     names_referenceTop = [names_referenceTop;strcat([pre_referenceTop,num2str(k,'%02d')])];
% end
% 
% pre_referenceRight = 'RR';
% names_referenceRight = {};
% 
% for k = 1:length(AllSides)
%     names_referenceRight = [names_referenceRight;strcat([pre_referenceRight,num2str(k,'%02d')])];
% end

% fprintf('*Nset, nset=AA-1, instance=PART-1-1\n');
% fprintf(num2str(100001));fprintf(',\n');
% fprintf('*Nset, nset=AA-2, instance=PART-1-1\n');
% fprintf(num2str(100002));fprintf(',\n');
% fprintf('*Nset, nset=AA-3, instance=PART-1-1\n');
% fprintf(num2str(100003));fprintf(',\n');
% fprintf('*Nset, nset=AA-4, instance=PART-1-1\n');
% fprintf(num2str(100004));fprintf(',\n');

fprintf('*Node\n');
node_TEXT    = '%d, %d, %d, %d\n';
fprintf(node_TEXT,1,unitCellSize*nnx,unitCellSize*nny,unitCellSize*nnz);

fprintf('*Node\n');
node_TEXT    = '%d, %d, %d, %d\n';
fprintf(node_TEXT,2,unitCellSize*nnx,unitCellSize*nny+50,unitCellSize*nnz);

fprintf('*Node\n');
node_TEXT    = '%d, %d, %d, %d\n';
fprintf(node_TEXT,3,unitCellSize*nnx+50,unitCellSize*nny,unitCellSize*nnz);

fprintf('*Node\n');
node_TEXT    = '%d, %d, %d, %d\n';
fprintf(node_TEXT,4,unitCellSize*nnx,unitCellSize*nny,unitCellSize*nnz+50);

fprintf('*Nset, nset=m1\n');
node_TEXT    = '%d\n';
fprintf(node_TEXT,1);

fprintf('*Nset, nset=n2\n');
node_TEXT    = '%d\n';
fprintf(node_TEXT,2);

fprintf('*Nset, nset=n1\n');
node_TEXT    = '%d\n';
fprintf(node_TEXT,3);

fprintf('*Nset, nset=n3\n');
node_TEXT    = '%d\n';
fprintf(node_TEXT,4);

% for j = 1:length(names_top)
%     TEXT =  names_top(j) ;
%     fprintf('*Nset, nset=');
%     fprintf(char(TEXT));
%     fprintf(', instance=PART-1-1\n');
%     fprintf(num2str(AllSides(j,1)));
%     fprintf(',\n');
%     
%     TEXT =  names_bot(j) ;
%     fprintf('*Nset, nset=');
%     fprintf(char(TEXT));
%     fprintf(', instance=PART-1-1\n');
%     fprintf(num2str(AllSides(j,2)));
%     fprintf(',\n');
%     
%     TEXT =  names_left(j) ;
%     fprintf('*Nset, nset=');
%     fprintf(char(TEXT));
%     fprintf(', instance=PART-1-1\n');
%     fprintf(num2str(AllSides(j,3)));
%     fprintf(',\n');
%     
%     TEXT =  names_right(j) ;
%     fprintf('*Nset, nset=');
%     fprintf(char(TEXT));
%     fprintf(', instance=PART-1-1\n');
%     fprintf(num2str(AllSides(j,4)));
%     fprintf(',\n');
%     
%     TEXT =  names_referenceTop(j) ;
%     fprintf('*Nset, nset=');
%     fprintf(char(TEXT));
%     fprintf(', instance=PART-1-1\n');
%     fprintf(num2str(500000+(2*j-1)));
%     fprintf(',\n');
% 
%     TEXT =  names_referenceRight(j) ;
%     fprintf('*Nset, nset=');
%     fprintf(char(TEXT));
%     fprintf(', instance=PART-1-1\n');
%     fprintf(num2str(500000+2*j));
%     fprintf(',\n');
% end

fprintf('*Nset, nset=TOP');fprintf(', instance=PART-1-1\n');count=0;
for i = 1:length(topNodes)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(topNodes(i)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');

fprintf('*Nset, nset=BOTTOM');fprintf(', instance=PART-1-1\n');count=0;
for i = 1:length(bottomNodes)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(bottomNodes(i)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');

fprintf('*Nset, nset=LEFT');fprintf(', instance=PART-1-1\n');count=0;
for i = 1:length(leftNodes)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(leftNodes(i)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');

fprintf('*Nset, nset=RIGHT');fprintf(', instance=PART-1-1\n');count=0;
for i = 1:length(rightNodes)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(rightNodes(i)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');

fprintf('*Nset, nset=FRONT');fprintf(', instance=PART-1-1\n');count=0;
for i = 1:length(frontNodes)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(frontNodes(i)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');

fprintf('*Nset, nset=BACK');fprintf(', instance=PART-1-1\n');count=0;
for i = 1:length(backNodes)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(backNodes(i)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');

fprintf('*Surface, type=NODE, name=RIGHT_CNS_, internal\n');
fprintf('RIGHT');fprintf(', ');fprintf('1\n');

fprintf('*Surface, type=NODE, name=TOP_CNS_, internal\n');
fprintf('TOP');fprintf(', ');fprintf('1\n');

fprintf('** Constraint: eqX1\n');
fprintf('*Equation\n');
fprintf('2\n');
fprintf('RIGHT');fprintf(', 1, 1.\n');
fprintf('m1');fprintf(', 1, -1.\n');

fprintf('** Constraint: eqY1\n');
fprintf('*Equation\n');
fprintf('2\n');
fprintf('TOP');fprintf(', 2, 1.\n');
fprintf('m1');fprintf(', 2, -1.\n');

fprintf('** Constraint: eqZ1\n');
fprintf('*Equation\n');
fprintf('2\n');
fprintf('FRONT');fprintf(', 3, 1.\n');
fprintf('m1');fprintf(', 3, -1.\n');

fprintf('** Constraint: eqX2\n');
fprintf('*Equation\n');
fprintf('2\n');
fprintf('LEFT');fprintf(', 1, -1.\n');
fprintf('m1');fprintf(', 1, -1.\n');

fprintf('** Constraint: eqY2\n');
fprintf('*Equation\n');
fprintf('2\n');
fprintf('BOTTOM');fprintf(', 2, -1.\n');
fprintf('m1');fprintf(', 2, -1.\n');

fprintf('** Constraint: eqZ2\n');
fprintf('*Equation\n');
fprintf('2\n');
fprintf('BACK');fprintf(', 3, -1.\n');
fprintf('m1');fprintf(', 3, -1.\n');

fprintf('*MPC, user, mode=DOF\n');
fprintf('1, n1, m1, n2\n');
fprintf('1, n3, m1, n2\n');

fprintf('*Spring, elset=Springs/Dashpots-1-spring\n');
fprintf('\n');
fprintf(num2str(SpringStiffness));fprintf('\n');
fprintf('*Element, type=SpringA, elset=Springs/Dashpots-1-spring\n');
fprintf('1, 1, 2\n');
fprintf('2, 1, 3\n');
fprintf('3, 1, 4\n');


% for k = 1:length(AllSides)
%     if (k~=1 && k~=(nnx+1))
%         fprintf('** Constraint: Constraint-DoF1_V');
%         fprintf(num2str(k));
%         fprintf(',\n');
%         fprintf('*Equation\n');
%         fprintf('3\n');
%         TEXT1 =  names_top(k) ;
%         TEXT2 =  names_bot(k) ;
%         TEXT3 =  names_referenceTop(k) ;
%         fprintf(char(TEXT1));fprintf(', 1, 1.\n');
%         fprintf(char(TEXT2));fprintf(', 1, -1.\n');
%         fprintf(char(TEXT3));fprintf(', 1, -1.\n');
%         %fprintf('AA-1, 1, -1.\n');
%         
%         fprintf('** Constraint: Constraint-DoF2_V');
%         fprintf(num2str(k));
%         fprintf(',\n');
%         fprintf('*Equation\n');
%         fprintf('3\n');
%         TEXT1 =  names_top(k) ;
%         TEXT2 =  names_bot(k) ;
%         TEXT3 =  names_referenceTop(k) ;
%         fprintf(char(TEXT1));fprintf(', 2, 1.\n');
%         fprintf(char(TEXT2));fprintf(', 2, -1.\n');
%         fprintf(char(TEXT3));fprintf(', 2, -1.\n');
%         %fprintf('AA-1, 2, -1.\n');
%         
%         fprintf('** Constraint: Constraint-DoF1_H');
%         fprintf(num2str(k));
%         fprintf(',\n');
%         fprintf('*Equation\n');
%         fprintf('3\n');
%         TEXT1 =  names_left(k) ;
%         TEXT2 =  names_right(k) ;
%         TEXT3 =  names_referenceRight(k) ;
%         fprintf(char(TEXT1));fprintf(', 1, -1.\n');
%         fprintf(char(TEXT2));fprintf(', 1, 1.\n');
%         fprintf(char(TEXT3));fprintf(', 1, -1.\n');
%         %fprintf('AA-4, 1, -1.\n');
%         
%         fprintf('** Constraint: Constraint-DoF2_H');
%         fprintf(num2str(k));
%         fprintf(',\n');
%         fprintf('*Equation\n');
%         fprintf('3\n');
%         TEXT1 =  names_left(k) ;
%         TEXT2 =  names_right(k) ;
%         TEXT3 =  names_referenceRight(k) ;
%         fprintf(char(TEXT1));fprintf(', 2, -1.\n');
%         fprintf(char(TEXT2));fprintf(', 2, 1.\n');
%         fprintf(char(TEXT3));fprintf(', 2, -1.\n');
%         %fprintf('AA-4, 2, -1.\n');
%     end
% end

% fprintf('** Constraint: Constraint-DB');
% fprintf(num2str(k));
% fprintf(',\n');
% fprintf('*Equation\n');
% fprintf('3\n');
% TEXT1 =  names_top(1) ;
% TEXT2 =  names_bot(nnx+1) ;
% TEXT3 =  names_referenceRight(nnx+1) ;
% fprintf(char(TEXT1));fprintf(', 1, -1.\n');
% fprintf(char(TEXT2));fprintf(', 1, 1.\n');
% fprintf(char(TEXT3));fprintf(', 1, -1.\n');
% %fprintf('AA-2, 1, -1.\n');
% 
% fprintf('** Constraint: Constraint-DB');
% fprintf(num2str(k));
% fprintf(',\n');
% fprintf('*Equation\n');
% fprintf('3\n');
% TEXT1 =  names_top(1) ;
% TEXT2 =  names_bot(nnx+1) ;
% TEXT3 =  names_referenceTop(1) ;
% fprintf(char(TEXT1));fprintf(', 2, -1.\n');
% fprintf(char(TEXT2));fprintf(', 2, 1.\n');
% fprintf(char(TEXT3));fprintf(', 2, 1.\n');
% %fprintf('AA-2, 2, -1.\n');
% 
% fprintf('** Constraint: Constraint-AC');
% fprintf(num2str(k));
% fprintf(',\n');
% fprintf('*Equation\n');
% fprintf('3\n');
% TEXT1 =  names_left(1) ;
% TEXT2 =  names_right(nnx+1) ;
% TEXT3 =  names_referenceRight(1) ;
% fprintf(char(TEXT1));fprintf(', 1, -1.\n');
% fprintf(char(TEXT2));fprintf(', 1, 1.\n');
% fprintf(char(TEXT3));fprintf(', 1, -1.\n');
% %fprintf('AA-3, 1, -1.\n');
% 
% fprintf('** Constraint: Constraint-AC');
% fprintf(num2str(k));
% fprintf(',\n');
% fprintf('*Equation\n');
% fprintf('3\n');
% TEXT1 =  names_left(1) ;
% TEXT2 =  names_right(nnx+1) ;
% TEXT3 =  names_referenceTop(nnx+1) ;
% fprintf(char(TEXT1));fprintf(', 2, -1.\n');
% fprintf(char(TEXT2));fprintf(', 2, 1.\n');
% fprintf(char(TEXT3));fprintf(', 2, -1.\n');
% %fprintf('AA-3, 2, -1.\n');

for i = indexEndAssembly(1):indexBoundary(1)
    TEXT = tlines(i);
    fprintf(char(TEXT));
    fprintf('\n');
end

% fprintf('** Name: BC-1 Type: Displacement/Rotation');fprintf('\n');
% fprintf('*Boundary');fprintf('\n');
% fprintf('LEFT, 1, 1, ');fprintf('\n');
% 
% fprintf('** Name: BC-2 Type: Displacement/Rotation');fprintf('\n');
% fprintf('*Boundary');fprintf('\n');
% fprintf('BOTTOM, 2, 2, ');fprintf('\n');

fprintf('** Name: BC-1 Type: Displacement/Rotation');fprintf('\n');
fprintf('*Boundary');fprintf('\n');
fprintf('n2, 2, 2, ');fprintf(num2str(maxDisp));fprintf('\n');

% fprintf('** Name: BC-2 Type: Displacement/Rotation');fprintf('\n');
% fprintf('*Boundary');fprintf('\n');
% fprintf('LEFT, 1, 1, ');fprintf('\n');
% 
% fprintf('** Name: BC-3 Type: Displacement/Rotation');fprintf('\n');
% fprintf('*Boundary');fprintf('\n');
% fprintf('BOTTOM, 2, 2, ');fprintf('\n');

% fprintf('** Name: BC-3 Type: Displacement/Rotation');fprintf('\n');
% fprintf('*Boundary, amplitude=AMP-1');fprintf('\n');
% fprintf('cornerDB, 1, 1, ');fprintf(num2str(loadCase(value,1)));fprintf('\n');
% fprintf('cornerDB, 2, 2, ');fprintf(num2str(-1*loadCase(value,2)));fprintf('\n');
% 
% fprintf('** Name: BC-4 Type: Displacement/Rotation');fprintf('\n');
% fprintf('*Boundary, amplitude=AMP-1');fprintf('\n');
% fprintf('cornerAC, 1, 1, ');fprintf(num2str(loadCase(value,1)));fprintf('\n');
% fprintf('cornerAC, 2, 2, ');fprintf(num2str(loadCase(value,2)));fprintf('\n');

for i = indexOutPutRequest(1):indexEnd(1)
    TEXT = tlines(i);
    fprintf(char(TEXT));
    fprintf('\n');
end

diary off