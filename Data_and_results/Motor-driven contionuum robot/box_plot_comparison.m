clear all
box2 = [];

load data/CDR_error_comparison.mat
start_idx=1;
Error = KMPC_ideal_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];

Error = KLQR_ideal_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];


Error = KILC1_ideal_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];
Error = KILC5_ideal_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];
KILC5_ideal_error = Error;


Error = KMPC_fault_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 );
box2 = [box2, Euclidean'];

Error = KLQR_fault_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 );
box2 = [box2, Euclidean'];

Error = KILC1_fault_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 );
box2 = [box2, Euclidean'];
Error = KILC5_fault_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 );
box2 = [box2, Euclidean'];

Error = KMPC_input_disturbance_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];

Error = KLQR_input_disturbance_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];

Error = KILC1_input_disturbance_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];
Error = KILC5_input_disturbance_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];


Error = KMPC_obstacle_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];

Error = KLQR_obstacle_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];

Error = KILC1_obstacle_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];
Error = KILC5_obstacle_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];


Error = KMPC_generalizability_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];

Error = KLQR_generalizability_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];

Error = KILC1_generalizability_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];
Error = KILC5_generalizability_error;
Euclidean = sqrt(Error(1,start_idx:end).^2 + Error(2,start_idx:end).^2 + Error(3,start_idx:end).^2);
box2 = [box2, Euclidean'];



c_map2 = get_color_map();

figure;
fontsize = 10;
interval = 1;
h1=boxplot(box2(:,1:interval:end),1:20,'OutlierSize',0.1,'symbol','','colors','k');hold on;
set(gca,'ylim',[0,15]);
set(gca,'FontSize',fontsize,'xticklabel','','XTickLabelRotation',45); 
ylabel("Error (mm)",'fontsize',fontsize);
% xlabel("Test",'fontsize',fontsize,'fontname','times new roman');
% title('Robot 1 tracking error');
% h=boxplot(MPG,Origin,'color','k');
h_box=findobj(gca,'Tag','Box');
h_M = findobj(gca,'Tag','Median');
 for j=1:length(h_box)
     h_M(j).Color='k';
     uistack(patch(get(h_box(j),'XData'),get(h_box(j),'YData'),c_map2(end-j+1,:),'FaceAlpha',1,'linewidth',1.5),'bottom');
%     p(j)=patch(get(h_box(j),'XData'),get(h_box(j),'YData'),c_map2(end-j+1,:),'FaceAlpha',1,'linewidth',1.5);
%     uistack(p(j),'bottom');
 end
 legend('KMPC','KLQR','KILC (k=1)', 'KILC (k=5)','fontsize',fontsize,'numcolumns',4);
% set(gca, h,'linewidth',1)






y = [KMPC_cost, KLQR_cost, KILC_cost1]*1000; % from s to ms

figure;
hold on;


color1 = [0 1 0]; 
color2 = [1 0 0]; 

for i = 1:length(y)

x_vertex = [i-0.2, i-0.2, i+0.2, i+0.2];
y_vertex = [0, y(i), y(i), 0];


z = [0, y(i), y(i), 0]; 
C = [linspace(color1(1), color2(1), 100)', ...
linspace(color1(2), color2(2), 100)', ...
linspace(color1(3), color2(3), 100)']; 

p = patch(x_vertex, y_vertex, z, 'FaceColor', 'interp', ...
'EdgeColor', 'none');

colormap(C);
end


xlim([0.5, length(y) + 0.5]);
ylim([0, max(y) + 1]);


% colorbar;
hold off;
box on;
% grid on;
ylabel('Time cost (ms)','fontsize',fontsize);
set(gca,'FontSize',fontsize,'xticklabel',{'','KMPC', '', 'KLQR', '', 'KILC'},'XTickLabelRotation',0); 



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
    
    %157,180,206;
    249,192,138;
    237,161,164;
    179,216,213;
    164,203,158;
    
    %157,180,206;
    249,192,138;
    237,161,164;
    179,216,213;
    164,203,158;

    %157,180,206;
    249,192,138;
    237,161,164;
    179,216,213;
    164,203,158;

    %157,180,206;
    249,192,138;
    237,161,164;
    179,216,213;
    164,203,158;

    %157,180,206;
    249,192,138;
    237,161,164;
    179,216,213;
    164,203,158;
    ]/255;
end