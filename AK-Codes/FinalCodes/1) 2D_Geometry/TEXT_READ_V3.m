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
ID = strfind(tlines,'** CONTROLS');
indexControl = find(not(cellfun('isempty',ID)));
ID = strfind(tlines,'*End Step');
indexEnd = find(not(cellfun('isempty',ID)));

diary(FileName)
for i = 1:indexNode(1)
    TEXT = tlines(i);
    fprintf(char(TEXT));
    fprintf('\n');
end
for i = 1:length(nodes)
    node_TEXT    = '%d, %f, %f\n';
    fprintf(node_TEXT,nodes(i,1),nodes(i,2),nodes(i,3));
end
% for i = 1:length(AllSides)
%     %fprintf('*Node\n');
%     node_TEXT    = '%d, %d, %d\n';
%     fprintf(node_TEXT,500000+(2*i-1),nodes(AllSides(i,1),2), unitCellSize*nnx+50);
%     fprintf(node_TEXT,500000+2*i    ,unitCellSize*nny+50           ,nodes(AllSides(i,4),3));
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

pre_top = 't';
names_top = {};

for k = 1:length(AllSides)
    names_top = [names_top;strcat([pre_top,num2str(k,'%02d')])];
end

pre_bot = 'b';
names_bot = {};

for k = 1:length(AllSides)
    names_bot = [names_bot;strcat([pre_bot,num2str(k,'%02d')])];
end

pre_right = 'r';
names_right = {};

for k = 1:length(AllSides)
    names_right = [names_right;strcat([pre_right,num2str(k,'%02d')])];
end

pre_left = 'l';
names_left = {};

for k = 1:length(AllSides)
    names_left = [names_left;strcat([pre_left,num2str(k,'%02d')])];
end

pre_referenceTop = 'RT';
names_referenceTop = {};

for k = 1:length(AllSides)
    names_referenceTop = [names_referenceTop;strcat([pre_referenceTop,num2str(k,'%02d')])];
end

pre_referenceRight = 'RR';
names_referenceRight = {};

for k = 1:length(AllSides)
    names_referenceRight = [names_referenceRight;strcat([pre_referenceRight,num2str(k,'%02d')])];
end


fprintf('*Node\n');
node_TEXT    = '%d, %d, %d, %d\n';
fprintf(node_TEXT,1,unitCellSize*nnx+50,unitCellSize*nny+50,0.);

fprintf('*Node\n');
node_TEXT    = '%d, %d, %d, %d\n';
fprintf(node_TEXT,2,unitCellSize*nnx+50,unitCellSize*nny+150,0.);

fprintf('*Node\n');
node_TEXT    = '%d, %d, %d, %d\n';
fprintf(node_TEXT,3,unitCellSize*nnx+150,unitCellSize*nny+50,0.);

fprintf('*Nset, nset=m1\n');
node_TEXT    = '%d\n';
fprintf(node_TEXT,1);

fprintf('*Nset, nset=n2\n');
node_TEXT    = '%d\n';
fprintf(node_TEXT,2);

fprintf('*Nset, nset=n1\n');
node_TEXT    = '%d\n';
fprintf(node_TEXT,3);


fprintf('*Nset, nset=TOP');fprintf(', instance=PART-1-1\n');count=0;
for i = 1:length(AllSides)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(AllSides(i,1)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');

fprintf('*Nset, nset=BOTTOM');fprintf(', instance=PART-1-1\n');count=0;
for i = 1:length(AllSides)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(AllSides(i,2)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');

fprintf('*Nset, nset=LEFT');fprintf(', instance=PART-1-1\n');count=0;
for i = 1:length(AllSides)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(AllSides(i,3)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');

fprintf('*Nset, nset=RIGHT');fprintf(', instance=PART-1-1\n');count=0;
for i = 1:length(AllSides)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(AllSides(i,4)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');


% fprintf('** Constraint: eqX1\n');
% fprintf('*Equation\n');
% fprintf('2\n');
% fprintf('AA_RIGHT');fprintf(', 1, 1.\n');
% fprintf('m1');fprintf(', 1, -1.\n');
% 
% fprintf('** Constraint: eqY1\n');
% fprintf('*Equation\n');
% fprintf('2\n');
% fprintf('AA_TOP');fprintf(', 2, 1.\n');
% fprintf('m1');fprintf(', 2, -1.\n');
% 
% fprintf('** Constraint: eqX2\n');
% fprintf('*Equation\n');
% fprintf('2\n');
% fprintf('LEFT');fprintf(', 1, -1.\n');
% fprintf('m1');fprintf(', 1, -1.\n');
% 
% fprintf('** Constraint: eqY2\n');
% fprintf('*Equation\n');
% fprintf('2\n');
% fprintf('BOTTOM');fprintf(', 2, -1.\n');
% fprintf('m1');fprintf(', 2, -1.\n');



fprintf('** Constraint: eqX1\n');
fprintf('*Equation\n');
fprintf('3\n');
fprintf('RIGHT');fprintf(', 1, 1.\n');
fprintf('LEFT');fprintf(', 1, -1.\n');
fprintf('m1');fprintf(', 1, -1.\n');

fprintf('** Constraint: eqY1\n');
fprintf('*Equation\n');
fprintf('3\n');
fprintf('TOP');fprintf(', 2, 1.\n');
fprintf('BOTTOM');fprintf(', 2, -1.\n');
fprintf('m1');fprintf(', 2, -1.\n');

fprintf('*MPC, user, mode=DOF\n');
fprintf('1, n1, m1, n2\n');

fprintf('*Spring, elset=Springs/Dashpots-1-spring\n');
fprintf('\n');
fprintf(num2str(SpringStiffness));fprintf('\n');
fprintf('*Element, type=SpringA, elset=Springs/Dashpots-1-spring\n');
fprintf('1, 1, 2\n');
fprintf('2, 1, 3\n');

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
% fprintf('AA_TOP, 1, 1, ');fprintf('\n');
% 
% fprintf('** Name: BC-3 Type: Displacement/Rotation');fprintf('\n');
% fprintf('*Boundary');fprintf('\n');
% fprintf('AA_RIGHT, 2, 2, ');fprintf('\n');

% fprintf('** Name: BC-3 Type: Displacement/Rotation');fprintf('\n');
% fprintf('*Boundary, amplitude=AMP-1');fprintf('\n');
% fprintf('cornerDB, 1, 1, ');fprintf(num2str(loadCase(value,1)));fprintf('\n');
% fprintf('cornerDB, 2, 2, ');fprintf(num2str(-1*loadCase(value,2)));fprintf('\n');
% 
% fprintf('** Name: BC-4 Type: Displacement/Rotation');fprintf('\n');
% fprintf('*Boundary, amplitude=AMP-1');fprintf('\n');
% fprintf('cornerAC, 1, 1, ');fprintf(num2str(loadCase(value,1)));fprintf('\n');
% fprintf('cornerAC, 2, 2, ');fprintf(num2str(loadCase(value,2)));fprintf('\n');

for i = indexControl(1):indexEnd(1)
    TEXT = tlines(i);
    fprintf(char(TEXT));
    fprintf('\n');
end

diary off