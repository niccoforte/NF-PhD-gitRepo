clear all
clc

nodes = [1 30 17.32;
    2 70 17.32;
    3 30 34.64;
    4 40 34.64;
    5 60 34.64;
    6 70 34.64;
    7 50 51.96;
    8 30 69.28;
    9 40 69.28;
    10 60 69.28;
    11 70 69.28;
    12 30 86.6;
    13 70 86.6];

element = [1 1 4;
    2 2 5;
    3 3 4;
    4 4 5;
    5 4 7;
    6 5 6;
    7 5 7;
    8 7 9;
    9 7 10;
    10 8 9;
    11 9 10;
    12 9 12;
    13 10 11;
    14 10 13];

nodes(:,2) = nodes(:,2)-30;
nodes(:,3) = nodes(:,3)-17.32;

RD = 0.1; L = max(nodes(:,2)); H = max(nodes(:,3));

len = zeros(length(element),1);
for ik = 1:length(element)
    x1 = nodes(element(ik,2),2);
    x2 = nodes(element(ik,3),2);
    y1 = nodes(element(ik,2),3);
    y2 = nodes(element(ik,3),3);
    len(ik,1) = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
end

% p = [4*RD (L+H-pi*sum(len)) 4*RD*(L*H)];
% dia_opt = roots(p);
% dia_est = 2*RD*4*L*H/(sum(len)*2*pi);
% diff_sqr = [(dia_opt(1)-dia_est)^2 (dia_opt(2)-dia_est)^2]; [minVal,index] = min(diff_sqr);
% rad = dia_opt(index)./2;
% rveVol = (L+2*rad)*(H+2*rad)*rad*2;

rad = 2*RD*L*H/(sum(len)*pi);
rveVol = (L)*(H)*rad*2;

[N] = nMatrix_new(element,nodes(:,2:3));
c_not = pi*rad*rad*(diag(len))./rveVol;
A = transpose(c_not*N)*N; C = inv(A);

syms S11 S22 S12
macroStress=[S11;S22;S12];
macroStrain=A^-1*macroStress;
microstrain=N*macroStrain;
microstress= microstrain;


FEA = [0.049777531	0.049979527
    0.025127698	0.049979528
    4.50008E-12	0.045183357
    -0.014525176	0.04379603
    -0.023530755	0.023626245
    -3.21E-02	3.22E-06
    -4.07E-02	-2.40E-02
    -4.98E-02	-5.00E-02
    -2.29E-02	-4.74E-02
    -4.50E-07	-4.52E-02
    1.43E-02	-4.33E-02
    2.35E-02	-2.36E-02
    3.26E-02	3.28E-08
    4.01E-02	2.25E-02
    0.049777	0.049979];

r = 0.05;
F = 0.5*[400 - 900 + 1/(r^2)];
G = 0.5*[900 - 400 + 1/(r^2)];
H = 0.5*[900 + 400 - 1/(r^2)];



hold on; box on;
h2 = plot(FEA(:,1),FEA(:,2),'--or','LineWidth',1.5);
% plot(FEA(:,1),FEA(:,2),'-r','LineWidth',1.5)
zfun = @(x,y) 0*x + 20*y;
zhandle = fcontour(zfun);
zhandle.LevelList = 1;
zhandle.LineWidth = 1.5;
zhandle.XRange = [-0.01666 0.05];
zhandle.YRange = [-0.01666 0.05];

zfun = @(x,y) 30*x - 10*y;
zhandle = fcontour(zfun);
zhandle.LevelList = 1;
zhandle.LineWidth = 1.5;
zhandle.XRange = [0.01666 0.05];
zhandle.YRange = [-0.05 0.05];

zfun = @(x,y) 0*x - 20*y;
zhandle = fcontour(zfun);
zhandle.LevelList = 1;
zhandle.LineWidth = 1.5;
zhandle.XRange = [-0.05 0.01666];
% zhandle.YRange = [-0.01666 0.05];

zfun = @(x,y) -30*x + 10*y;
zhandle = fcontour(zfun);
zhandle.LevelList = 1;
zhandle.LineWidth = 1.5;
zhandle.YRange = [-0.05 0.05];

zfun1 = @(x,y) sqrt((F+H)*x.^2 + (G+H)*y.^2 - 2*H*x*y);
zhandle1 = fcontour(zfun1);
zhandle1.LevelList = 1;
zhandle1.LineWidth = 1.5;
zhandle1.LineColor = 'blue';

xlim([-0.1 0.1])
ylim([-0.1 0.1])
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
xlabel([char(931) '_{11}/' char(963) '_{y}'],'FontSize',14)
ylabel([char(931) '_{22}/' char(963) '_{y}'],'FontSize',14)
legend([zhandle h2],'analytical solution','FE calculation')
% legend([zhandle h2 zhandle1],'analytical solution','FE calculation', 'Hill''s Criteria')
    
for i = 1:size(element,1)
    coord1 = nodes(element(i,2),2:3);
    coord2 = nodes(element(i,3),2:3);
    figure(2000)
    hold on
%     axis square
    axis off
%     if ismember(i,bElem(:,1))
%         plot([coord1(1);coord2(1)],[coord1(2);coord2(2)],'-r','LineWidth',2.0)
%     else
        plot([coord1(1);coord2(1)],[coord1(2);coord2(2)],'-b','LineWidth',2.5)
%     end
end
figure(2000)
hold on
% axis square
axis off
set(gcf,'color','w');