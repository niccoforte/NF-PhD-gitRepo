fid=fopen('Tension_v3_exp.inp');
       
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

for i = indexSection(1):indexSection(1)
    TEXT = tlines(i);
    fprintf(char(TEXT));
    fprintf('\n');
end

if (strcmpi(crossSec,'circ'))
    fprintf('*Beam Section, elset=SET-2, material=ALUM, temperature=GRADIENTS, section=CIRC');
    fprintf('\n');
    fprintf(num2str(rad));fprintf('\n');
elseif (strcmpi(crossSec,'rect'))
    fprintf('*Beam Section, elset=SET-2, material=ALUM, temperature=GRADIENTS, section=RECT');
    fprintf('\n');
    fprintf(num2str(rad1));fprintf(', ');fprintf(num2str(rad2));fprintf('\n');
end


for i = indexSection(1)+3:indexNode(2)-1
    TEXT = tlines(i);
    fprintf(char(TEXT));
    fprintf('\n');
end


fprintf('*Node\n');
node_TEXT    = '%d, %d, %d, %d\n';
fprintf(node_TEXT,1,L+150,H+150,0.);

if (strcmpi(loadingType,'ten'))
    fprintf('*Node\n');
    node_TEXT    = '%d, %d, %d, %d\n';
    fprintf(node_TEXT,2,L+150,H+250,0.);
elseif (strcmpi(loadingType,'bicom'))
    fprintf('*Node\n');
    node_TEXT    = '%d, %d, %d, %d\n';
    fprintf(node_TEXT,2,L+150,H+50,0.);
elseif (strcmpi(loadingType,'com'))
    fprintf('*Node\n');
    node_TEXT    = '%d, %d, %d, %d\n';
    fprintf(node_TEXT,2,L+150,H+250,0.);
end


if (strcmpi(loadingType,'ten'))
    fprintf('*Node\n');
    node_TEXT    = '%d, %d, %d, %d\n';
    fprintf(node_TEXT,3,L+250,H+150,0.);
elseif (strcmpi(loadingType,'bicom'))
    fprintf('*Node\n');
    node_TEXT    = '%d, %d, %d, %d\n';
    fprintf(node_TEXT,3,L+50,H+150,0.);
elseif (strcmpi(loadingType,'com'))
    fprintf('*Node\n');
    node_TEXT    = '%d, %d, %d, %d\n';
    fprintf(node_TEXT,3,L+50,H+150,0.);
end



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
for i = 1:length(topNodes)
    count = count+1;
%     if (i==(nnx+1))
%         
%     else
        fprintf(num2str(topNodes(i,1)));fprintf(', ');
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
        fprintf(num2str(bottomNodes(i,1)));fprintf(', ');
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
        fprintf(num2str(leftNodes(i,1)));fprintf(', ');
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
        fprintf(num2str(rightNodes(i,1)));fprintf(', ');
%     end
    if count==10
        fprintf('\n');
        count=0;
    end
end
fprintf('\n');


% fprintf('** Constraint: eqX1\n');
% fprintf('*Equation\n');
% fprintf('3\n');
% fprintf('RIGHT');fprintf(', 1, 1.\n');
% fprintf('LEFT');fprintf(', 1, -1.\n');
% fprintf('m1');fprintf(', 1, -1.\n');
% 
% fprintf('** Constraint: eqY1\n');
% fprintf('*Equation\n');
% fprintf('3\n');
% fprintf('TOP');fprintf(', 2, 1.\n');
% fprintf('BOTTOM');fprintf(', 2, -1.\n');
% fprintf('m1');fprintf(', 2, -1.\n');

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


fprintf('*Element, type=MASS, elset=M1_Inertia-1_\n');
fprintf('1, 1\n');
fprintf('*Mass, elset=M1_Inertia-1_\n');
fprintf('1e-8, \n');

fprintf('*Element, type=MASS, elset=N1_Inertia-1_\n');
fprintf('2, 3\n');
fprintf('*Mass, elset=N1_Inertia-1_\n');
fprintf('1, \n');

fprintf('*Element, type=MASS, elset=N2_Inertia-1_\n');
fprintf('3, 2\n');
fprintf('*Mass, elset=N2_Inertia-1_\n');
fprintf('1, \n');

fprintf('*Spring, elset=SPRINGS/DASHPOTS-1-SPRING1\n');
fprintf('\n');
fprintf(num2str(SpringStiffness));fprintf('\n');
fprintf('*Element, type=SpringA, elset=SPRINGS/DASHPOTS-1-SPRING1\n');
fprintf('4, 1, 2\n');


fprintf('*Spring, elset=SPRINGS/DASHPOTS-1-SPRING2\n');
fprintf('\n');
fprintf(num2str(abs(rho(value))*SpringStiffness*(L/H)));fprintf('\n');
fprintf('*Element, type=SpringA, elset=SPRINGS/DASHPOTS-1-SPRING2\n');
fprintf('5, 1, 3\n');

for i = indexEndAssembly(1):indexBoundary(1)
    TEXT = tlines(i);
    fprintf(char(TEXT));
    fprintf('\n');
end

if (strcmpi(loadingType,'ten'))
    fprintf('** Name: Vel-BC-1 Type: Displacement/Rotation');fprintf('\n');
    fprintf('*Boundary, amplitude=AMP-1');fprintf('\n');
    fprintf('n2, 2, 2, ');fprintf(num2str(Amplitude*1));fprintf('\n');
    
    fprintf('** Name: Vel-BC-1 Type: Displacement/Rotation');fprintf('\n');
    fprintf('*Boundary, amplitude=AMP-1');fprintf('\n');
    fprintf('n1, 1, 1, ');fprintf(num2str(Amplitude*1));fprintf('\n');
elseif (strcmpi(loadingType,'bicom'))    
    fprintf('** Name: Vel-BC-1 Type: Displacement/Rotation');fprintf('\n');
    fprintf('*Boundary, amplitude=AMP-1');fprintf('\n');
    fprintf('n2, 2, 2, ');fprintf(num2str(-1*Amplitude));fprintf('\n');
    
    fprintf('** Name: Vel-BC-1 Type: Displacement/Rotation');fprintf('\n');
    fprintf('*Boundary, amplitude=AMP-1');fprintf('\n');
    fprintf('n1, 1, 1, ');fprintf(num2str(-1*Amplitude));fprintf('\n');
elseif (strcmpi(loadingType,'com'))
    fprintf('** Name: Vel-BC-1 Type: Displacement/Rotation');fprintf('\n');
    fprintf('*Boundary, amplitude=AMP-1');fprintf('\n');
    fprintf('n2, 2, 2, ');fprintf(num2str(Amplitude*1));fprintf('\n');
    
    fprintf('** Name: Vel-BC-1 Type: Displacement/Rotation');fprintf('\n');
    fprintf('*Boundary, amplitude=AMP-1');fprintf('\n');
    fprintf('n1, 1, 1, ');fprintf(num2str(-1*Amplitude));fprintf('\n');
end

for i = indexControl(1):indexEnd(1)
    TEXT = tlines(i);
    fprintf(char(TEXT));
    fprintf('\n');
end

diary off