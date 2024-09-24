% This file contains the MATLAB code for drawing a radar graph

clear all

% [ideal, fault, input_disturbance, obstacle, generalizability, no fault]
KMPC_RMSE=[2.02, 2.34, 7.21, 8.18, 4.76, 1.85];
KMPC_ME=  [4.59, 8.20, 10.26, 31.08, 7.78, 6.19];
KMPC_cost=4.65;
KLQR_RMSE=[6.28, 5.00, 8.54, 10.32, 7.20, 4.94];
KLQR_ME=  [10.54, 11.43, 14, 23.36, 12.96, 9.44];
KLQR_cost=0.27;
KILC1_RMSE=[1.16, 0.91, 1.26, 1.24, 1.18, 0.89];
KILC1_ME=  [3.07, 4.16, 2.21, 3.22, 2.97, 4.17];
KILC1_cost=0.32;
KILC5_RMSE=[0.97, 0.77, 1.14, 1.12, 0.96, 0.79];
KILC5_ME=  [2.66, 3.96, 3.63, 3.05, 2.75, 3.72];
KILC5_cost=0.3;

L = 260;
L1p = L*0.01;
L5p = L*0.1;
max_dege=1.0;
KMPC_index=[1-min((KMPC_RMSE(1))/L5p,1), 1-min((KMPC_RMSE(2)-KMPC_RMSE(6))/KMPC_RMSE(6),1), 1-min((KMPC_RMSE(3)-KMPC_RMSE(1))/(1*KMPC_RMSE(1)),1), 1-min((KMPC_RMSE(4)-KMPC_RMSE(1))/(1*KMPC_RMSE(1)),1), 1-min((KMPC_RMSE(5)-KMPC_RMSE(1))/(1*KMPC_RMSE(1)),1), 1-KMPC_cost/6];
KLQR_index=[1-min((KLQR_RMSE(1))/L5p,1), 1-min((KLQR_RMSE(2)-KLQR_RMSE(6))/KLQR_RMSE(6),1), 1-min((KLQR_RMSE(3)-KLQR_RMSE(1))/(1*KLQR_RMSE(1)),1), 1-min((KLQR_RMSE(4)-KLQR_RMSE(1))/(1*KLQR_RMSE(1)),1), 1-min((KLQR_RMSE(5)-KLQR_RMSE(1))/(1*KLQR_RMSE(1)),1),  1-KLQR_cost/6];
KILC1_index=[1-min((KILC1_RMSE(1))/L5p,1), 1-min((KILC1_RMSE(2)-KILC1_RMSE(6))/KILC1_RMSE(6),1), 1-min((KILC1_RMSE(3)-KILC1_RMSE(1))/(1*KILC1_RMSE(1)),1), 1-min((KILC1_RMSE(4)-KILC1_RMSE(1))/(1*KILC1_RMSE(1)),1), 1-min((KILC1_RMSE(5)-KILC1_RMSE(1))/(1*KILC1_RMSE(1)),1),  1-KILC1_cost/6];
KILC3_index=[1-min((KILC5_RMSE(1))/L5p,1), min(1-min((KILC5_RMSE(2)-KILC1_RMSE(6))/KILC1_RMSE(6),1),max_dege), min(1,1-min((KILC5_RMSE(3)-KILC1_RMSE(1))/(1*KILC1_RMSE(1)),max_dege)), min(1,1-min((KILC5_RMSE(4)-KILC1_RMSE(1))/(1*KILC1_RMSE(1)),max_dege)), min(1,1-min((KILC5_RMSE(5)-KILC1_RMSE(1))/(1*KILC1_RMSE(1)),max_dege)), 1-KILC5_cost/6];

% KMPC_index=[1-min((KMPC_RMSE(1)-L1p)/L5p,1), 1-min((KMPC_RMSE(2)-KMPC_RMSE(1))/L5p,1), 1-min((KMPC_RMSE(3)-KMPC_RMSE(1))/L5p,1), 1-KMPC_cost/6];
% KLQR_index=[1-min((KLQR_RMSE(1)-L1p)/L5p,1), 1-min((KLQR_RMSE(2)-KLQR_RMSE(1))/L5p,1), 1-min((KLQR_RMSE(3)-KLQR_RMSE(1))/L5p,1),  1-KLQR_cost/6];
% KILC1_index=[1-min((KILC1_RMSE(1)-L1p)/L5p,1), 1-min((KILC1_RMSE(2)-KILC1_RMSE(1))/L5p,1), 1-min((KILC1_RMSE(3)-KILC1_RMSE(1))/L5p,1),  1-KILC1_cost/6];
% KILC3_index=[1-min((KILC5_RMSE(1)-L1p)/L5p,1), min(1-min((KILC5_RMSE(2)-KILC1_RMSE(1))/L5p,1),1), min(1,1-min((KILC5_RMSE(3)-KILC1_RMSE(1))/L5p,1)), 1-KILC5_cost/6];


index_num=6; % set the number of indeces
rad=linspace(0,2*pi,index_num+1);
k=rad(1:length(rad)-1);

% set the percentage for all indeces
coef = [0.1,0.3,0.8,0.7,0.5]; % ranging [0,1]

color_map = get_color_map();


add_index_name=false;
figure
fontsize=10;
% fill the index polygon with color
coef = [coef;coef];
performance_KMPC = [sin(k);cos(k)].*KMPC_index;
performance_KLQR = [sin(k);cos(k)].*KLQR_index;
performance_KILC1 = [sin(k);cos(k)].*KILC1_index;
performance_KILC3 = [sin(k);cos(k)].*KILC3_index;
p4 = patch(performance_KILC3(1,:),performance_KILC3(2,:),1:length(k),'facecolor',color_map(end,:),'facealpha',1,'edgecolor',[0,162,64]/255,'linewidth',0.5);hold on;
p3 = patch(performance_KILC1(1,:),performance_KILC1(2,:),1:length(k),'facecolor',color_map(end-1,:),'facealpha',1,'edgecolor',[0,180,184]/255,'linewidth',0.5);
p2 = patch(performance_KLQR(1,:),performance_KLQR(2,:),1:length(k),'facecolor',color_map(end-2,:),'facealpha',0.8,'edgecolor',[254,88,100]/255,'linewidth',0.5);
p1 = patch(performance_KMPC(1,:),performance_KMPC(2,:),1:length(k),'facecolor',color_map(end-3,:),'facealpha',0.6,'edgecolor',[230,117,4]/255,'linewidth',0.5); 

% % draw vertexes
% scatter(performance_KILC3(1,:),performance_KILC3(2,:),40,'g','fill','linewidth',2);
% scatter(performance_KILC1(1,:),performance_KILC1(2,:),40,'c','fill','linewidth',2);
% scatter(performance_KLQR(1,:),performance_KLQR(2,:),40,'m','fill','linewidth',2);
% scatter(performance_KMPC(1,:),performance_KMPC(2,:),40,'y','fill','linewidth',2);
% draw the background polygon
level_num = 5;
for i = 1:level_num
    rate=1/level_num*i;
    points=[sin(k);cos(k)]*rate;
    if i==level_num
        % flag=0;
        % draw_polygon(points,color_map(3,:),flag);
    else
        flag=1;
        draw_polygon(points,color_map(3,:),flag);
    end
end

legend([p1,p2,p3,p4],'KMPC','KLQR','KILC (k=1)','KILC (k=5)','fontsize',fontsize,'numcolumns',4);

if add_index_name==true
    % add index name
    % fontsize = 14;
    index_name_pos = [sin(k);cos(k)]*1.2;
    index_name={'Accuracy','Fault tolerance','Input disturbance rejection','External disturbance rejection', 'Generalizability','Computational efficiency'};
    for i = 1:index_num
    text(index_name_pos(1,i),index_name_pos(2,i),index_name{i},'fontsize',fontsize);
    end
end

set(gca,'xlim',[-1.8,1.8],'ylim',[-1.8,1.8]);
axis equal;

function draw_polygon(points,color,flag)
% draw a polygon with vertexes contained in 'points'
% points = [x1,x2,x3,x4,x5,...; y1,y2,y3,y4,y5,...]
point=points;
point = [point, points(:,1)];
edge_color=color;
for i = 1:size(points,2)
    if flag==1
        p=plot(point(1,i:i+1), point(2,i:i+1),'--','color',edge_color,'linewidth',1);hold on
    else
        plot(point(1,i:i+1), point(2,i:i+1),'-k','linewidth',1);hold on
    end
    
        % plot([0,point(1,i)], [0,point(2,i)],'--','color',edge_color,'linewidth',1);
end
end

function color_map = get_color_map()
color_map=[113,57,72;
    227,115,139;
    249,213,221;
    215,198,202;
    73,83,115;
    140,165,234;
    201,203,214;
    220,228,250;
    121,91,52;
    252,180,98;
    218,206,194;
    255,232,106;
    63,101,98;
    123,196,197;
    197,209,210;
    217,238,238;
    
    157,180,206;
    249,192,138;
    237,161,164;
    179,216,213;
    164,203,158;
    
    157,180,206;
    249,192,138;
    237,161,164;
    179,216,213;
    164,203,158;
    ]/255;
end