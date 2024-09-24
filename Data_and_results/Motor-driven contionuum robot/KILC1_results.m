
% save  data/CDR_error_KILC.mat x_d1 x_d2 x_d3 x_d4 x_d5 x_k1 x_k2 x_k3 x_k4 x_k5 

clear all
load data/CDR_error_KILC.mat

fontsize = 13;
figure
plot3(x_d1(1,:), x_d1(2,:), x_d1(3,:), '-', 'color', [232, 104,39]/255, 'linewidth', 2); hold on;
plot3(x_k1(1,:), x_k1(2,:), x_k1(3,:), '--*', 'color', [19, 30,253]/255, 'linewidth', 1, 'MarkerIndices',1:20:600); hold on;
box on;
axis equal;
hold off;
xMin = -100; xMax = 40; yMin = 0; yMax = 140; zMin = 600; zMax = 700;
axis([xMin xMax yMin yMax zMin zMax]);
set(gca, 'xtick', xMin:20:xMax);
set(gca, 'ytick', yMin:20:yMax);
set(gca, 'ztick', zMin:20:zMax);
set(gca, 'FontSize', fontsize);
xlabel( '$X$ (mm)' , 'FontSize', fontsize, 'interpreter','latex' );
ylabel( '$Y$ (mm)' , 'FontSize', fontsize, 'interpreter','latex' );
zlabel( '$Z$ (mm)' , 'FontSize', fontsize, 'interpreter','latex' );
legend({'Reference trajectory' , 'Actual trajectory'}, 'FontSize', fontsize, 'NumColumns', 2);



figure
subplot(3, 1 , 1 );
fontsize=13;
t=0:0.1:60-0.1;
interval=1;
plot(t(1:interval:end), x_d1(1,1:interval:end), '-', 'linewidth',2);hold on;
plot(t(1:interval:end), x_k1(1,1:interval:end), ':*','linewidth',1 , 'MarkerIndices',1:20:600);
set(gca, 'FontSize', fontsize)
ylabel( 'X' , 'FontSize', fontsize, 'interpreter','latex', 'Rotation', pi/2 );
hold off;

subplot(3, 1 , 2 );
ylabel( 'Y' , 'FontSize', fontsize, 'interpreter','latex' , 'Rotation', pi/2);
hold on;
plot(t(1:interval:end), x_d1(2,1:interval:end), '-', 'linewidth',2);
plot(t(1:interval:end), x_k1(2,1:interval:end), ':*','linewidth',1 , 'MarkerIndices',1:20:600);
set(gca, 'FontSize', fontsize)
hold off;

subplot(3 , 1 , 3 );
ylabel( 'Z' , 'FontSize', fontsize, 'interpreter','latex', 'Rotation', pi/2 );
hold on;
plot(t(1:interval:end), x_d1(3,1:interval:end), '-', 'linewidth',2);
plot(t(1:interval:end), x_k1(3,1:interval:end), ':*','linewidth',1, 'MarkerIndices',1:20:600);
set(gca, 'FontSize', fontsize, 'ylim',[600,700]);
hold off;
legend({'Reference trajectory' , 'Actual trajectory'}, 'FontSize', fontsize, 'NumColumns', 2);




box = [];
start_idx=1;
error = x_d1-x_k1;
Euclidean = sqrt(error(1,start_idx:end).^2 + error(2,start_idx:end).^2 + error(3,start_idx:end).^2);
box = [box, Euclidean'];
error = x_d2-x_k2;
Euclidean = sqrt(error(1,start_idx:end).^2 + error(2,start_idx:end).^2 + error(3,start_idx:end).^2);
box = [box, Euclidean'];
error = x_d3-x_k3;
Euclidean = sqrt(error(1,start_idx:end).^2 + error(2,start_idx:end).^2 + error(3,start_idx:end).^2);
box = [box, Euclidean'];
error = x_d4-x_k4;
Euclidean = sqrt(error(1,start_idx:end).^2 + error(2,start_idx:end).^2 + error(3,start_idx:end).^2);
box = [box, Euclidean'];
error = x_d5-x_k5;
Euclidean = sqrt(error(1,start_idx:end).^2 + error(2,start_idx:end).^2 + error(3,start_idx:end).^2);
box = [box, Euclidean'];



c_map2 = [0.77, 0.18, 0.78;
          0.21, 0.33, 0.64;
          0.88, 0.17, 0.56;
          0.20, 0.69, 0.28;
          0.26, 0.15, 0.47;
          0.83, 0.27, 0.44;
          0.87, 0.85, 0.42;
          0.85, 0.51, 0.87;
          0.99, 0.62, 0.76;
          0.52, 0.43, 0.87;
          0.00, 0.68, 0.92;
          0.26, 0.45, 0.77;
          0.98, 0.75, 0.00;    
          0.72, 0.81, 0.76;
          0.77, 0.18, 0.78;
          0.28, 0.39, 0.44;
          0.22, 0.26, 0.24;
          0.64, 0.52, 0.64;
          0.87, 0.73, 0.78;
          0.94, 0.89, 0.85;
          0.85, 0.84, 0.86];

color1 = [13 255 88]/255; 
color2 = [255 157 13]/255; 
C = [linspace(color1(1), color2(1), 5)', ...
linspace(color1(2), color2(2), 5)', ...
linspace(color1(3), color2(3), 5)'];

figure;
fontsize = 15;
interval = 1;
h=boxplot(box(:,1:interval:end),1:5,'OutlierSize',0.1,'symbol','');grid on;
% set(gca,'FontSize',fontsize,'xticklabel',{'STC-ZNN','SyncTC','CI-SyncTC'},'XTickLabelRotation',45,'fontname','times new roman'); 
% xlabel("Test",'fontsize',fontsize,'fontname','times new roman');
ylabel("Error (mm)",'fontsize',fontsize);
% title('Robot 1 tracking error');
% h=boxplot(MPG,Origin,'color','k');
h_box=findobj(gca,'Tag','Box');
h_M = findobj(gca,'Tag','Median');
 for j=1:length(h_box)
     h_M(j).Color='k';
     uistack(patch(get(h_box(j),'XData'),get(h_box(j),'YData'),C(j,:),'FaceAlpha',1,'linewidth',1.5),'bottom');
 end
set(h,'linewidth',1)
set(gca,'ylim',[0,2.5], 'fontsize', 15);
xlabel('Iteration','fontsize',fontsize);