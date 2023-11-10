%clear all
close all
clc
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

pre_left = 't';
names_left = {};

for k = 1:length(AllSides)
    names_left = [names_left;strcat([pre_left,num2str(k,'%02d')])];
end


delete NAMES.txt
diary NAMES.txt
for j = 2:length(names_top)-1
    TEXT =  names_top(j) ;
    fprintf('*Nset, nset=');
    fprintf(char(TEXT));
    fprintf(', instance=PART-1-1\n');
    fprintf(num2str(AllSides(j,1)));
    fprintf(',\n');
    
    TEXT =  names_bot(j) ;
    fprintf('*Nset, nset=');
    fprintf(char(TEXT));
    fprintf(', instance=PART-1-1\n');
    fprintf(num2str(AllSides(j,2)));
    fprintf(',\n');
    
    TEXT =  names_left(j) ;
    fprintf('*Nset, nset=');
    fprintf(char(TEXT));
    fprintf(', instance=PART-1-1\n');
    fprintf(num2str(AllSides(j,3)));
    fprintf(',\n');
    
    TEXT =  names_right(j) ;
    fprintf('*Nset, nset=');
    fprintf(char(TEXT));
    fprintf(', instance=PART-1-1\n');
    fprintf(num2str(AllSides(j,4)));
    fprintf(',\n');
end

for k = 2:length(AllSides)-1
    fprintf('** Constraint: Constraint-DoF1_');
    fprintf(num2str(k));
    fprintf(',\n');
    fprintf('*Equation\n');
    fprintf('3\n');
    TEXT1 =  names_top(k) ;
    TEXT2 =  names_bot(k) ;
    fprintf(char(TEXT1));fprintf(', 1, 1.\n');
    fprintf(char(TEXT2));fprintf(', 1, -1.\n');
    fprintf('RP-right, 1, -1.\n');
    
    fprintf('** Constraint: Constraint-DoF2_');
    fprintf(num2str(k));
    fprintf(',\n');
    fprintf('*Equation\n');
    fprintf('3\n');
    TEXT1 =  names_top(k) ;
    TEXT2 =  names_bot(k) ;
    fprintf(char(TEXT1));fprintf(', 2, 1.\n');
    fprintf(char(TEXT2));fprintf(', 2, -1.\n');
    fprintf('RP-Top, 2, -1.\n');
end
diary off