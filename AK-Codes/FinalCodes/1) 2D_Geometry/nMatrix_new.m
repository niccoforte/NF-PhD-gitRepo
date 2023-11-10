function [N] = nMatrix_new(element,nodes)

% Calculating N matrix

%angle with horizontal axis
N = zeros(length(element),3);
for i = 1:length(element)
    x = [0 0];
    y = [20 0];
    a = nodes(element(i,2),:);
    b = nodes(element(i,3),:);
    difference = (atan((y(2)-x(2))/(y(1)-x(1))) - atan((b(2)-a(2))/(b(1)-a(1))));
    n1 = cos(difference);
    n2 = sin(difference);
    N(i,:) = [n1*n1 n2*n2 n1*n2];
end