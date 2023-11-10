clear all
close all
clc

nnx = 10;
nny = 10;
L = 4*nnx;
H = 4*nny;
fac = 0;

[nodes,nodesR] = node_new(L,H,nnx,nny,fac);

[element] = connectivity_new(L,H,nnx,nny,nodes,nodesR);

nodeNum = nodesR(:,1);
nodeX   = nodesR(:,2);
nodeY   = nodesR(:,3);

elemNum = element(:,1);
elem_1  = element(:,2);
elem_2  = element(:,3);

delete *.txt

diary Nodes.txt
for i = 1:length(nodeNum)
    node_TEXT    = '%d, %f, %f\n';
    fprintf(node_TEXT,nodeNum(i),nodeX(i),nodeY(i));
end
diary off

diary Elements.txt
for i = 1:length(elemNum)
    node_TEXT    = '%d, %d, %d\n';
    fprintf(node_TEXT,elemNum(i),elem_1(i),elem_2(i));
end
diary off


