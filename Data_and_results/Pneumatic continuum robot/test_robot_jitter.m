clear all;

load data/test_robot_jitter.mat;
addpath('functions');



Td = 180;
tau  = 0.1;
number = Td/tau+1;
% number=900;
interval = 1;
[all_themes, all_colors] = GetColors();
fontsize = 13;
t = 0:tau:Td;


p1 = polyfit(t,xa(1,:),50);
p2 = polyfit(t,xa(2,:),50);
p3 = polyfit(t,xa(3,:),50);

y1 = polyval(p1,t);
y2 = polyval(p2,t);
y3 = polyval(p3,t);




start = 1;
figure;

plot3(xa(1,start:interval:number), xa(2,start:interval:number), xa(3,start:interval:number),'-.o','color',[113, 57, 72]'/255);grid on;hold on; 

% tail = [75, 0, 380];
% head = [90, -5, 385];
% quiver3(tail(1), tail(2), tail(3), head(1)-tail(1), head(2)-tail(2), head(3)-tail(3), 'k','linewidth', 1);
% text(65, 5, 380, 'Large temporary noise', 'FontName', 'times new Roman','fontsize',18)
axis equal;
legend( 'Actual trajactory','Location','best', 'FontName', 'times new Roman','fontsize',fontsize);
hold off;
set(gca,'FontSize',fontsize, 'FontName', 'times new Roman','colororder', all_themes{6});
xlabel('$x$ (mm)', 'FontName', 'times new Roman','fontsize',fontsize,'interpreter','latex');
ylabel('$y$ (mm)', 'FontName', 'times new Roman','fontsize',fontsize,'interpreter','latex');
zlabel('$z$ (mm)', 'FontName', 'times new Roman','fontsize',fontsize,'interpreter','latex');
% savefig('results\Circle_50.fig');


number = 1801;
error1 = y1 - xa(1,:);
error2 = y2 - xa(2,:);
error3 = y3 - xa(3,:);
RMSE1 = round(sqrt(sum(error1.^2 )/number), 4);
RMSE2 = round(sqrt(sum(error2.^2 )/number), 4);
RMSE3 = round(sqrt(sum(error3.^2 )/number), 4);

% fontsize = 26;
figure
subplot(2,2,2);
fill([t,fliplr(t)], [y1-max(error1), fliplr(y1+max(error1))],[249, 213, 221]/255); hold on
fill([t,fliplr(t)], [y1-RMSE1, fliplr(y1+RMSE1)],[227, 115, 139]/255);
plot(t,xa(1,:),  'linewidth', 1, 'color',[113, 57, 72]/255); 
set(gca,'FontSize',fontsize, 'FontName', 'times new Roman');
yMin = -100;
yMax = 0;
xMin = 0;
xMax = 90;
axis([xMin xMax yMin yMax]);
set(gca, 'xtick', xMin:30:xMax);
set(gca, 'ytick', yMin:20:yMax);
xlabel('Time (s)', 'FontName', 'times new Roman','fontsize',fontsize );
ylabel('$X$ (mm)', 'FontName', 'times new Roman','fontsize',fontsize,'interpreter','latex');
legend('Maximum jitter range', 'Mean jitter range', 'End-effector position', 'FontName', 'times new Roman','fontsize',fontsize)

subplot(2,2,3);
fill([t,fliplr(t)], [y2-max(error2), fliplr(y2+max(error2))],[249, 213, 221]/255); hold on
fill([t,fliplr(t)], [y2-RMSE2, fliplr(y2+RMSE2)],[227, 115, 139]/255);
plot(t,xa(2,:),  'linewidth', 1, 'color',[113, 57, 72]/255);
set(gca,'FontSize',fontsize, 'FontName', 'times new Roman');
yMin = 40;
yMax = 80;
xMin = 0;
xMax = 90;
axis([xMin xMax yMin yMax]);
set(gca, 'xtick', xMin:30:xMax);
set(gca, 'ytick', yMin:10:yMax);
xlabel('Time (s)', 'FontName', 'times new Roman','fontsize',fontsize );
ylabel('$Y$ (mm)', 'FontName', 'times new Roman','fontsize',fontsize,'interpreter','latex');
legend('Maximum jitter range', 'Mean jitter range', 'End-effector position', 'FontName', 'times new Roman','fontsize',fontsize)

subplot(2,2,4);
fill([t,fliplr(t)], [y3-max(error3), fliplr(y3+max(error3))],[249, 213, 221]/255); hold on
fill([t,fliplr(t)], [y3-RMSE3, fliplr(y3+RMSE3)],[227, 115, 139]/255);
plot(t,xa(3,:),  'linewidth', 1, 'color',[113, 57, 72]/255);
yMin = 1000;
yMax = 1200;
xMin = 0;
xMax = 90;
axis([xMin xMax yMin yMax]);
set(gca, 'xtick', xMin:30:xMax);
set(gca, 'ytick', yMin:50:yMax);
set(gca,'FontSize',fontsize, 'FontName', 'times new Roman');
xlabel('Time (s)', 'FontName', 'times new Roman','fontsize',fontsize );
ylabel('$Z$ (mm)', 'FontName', 'times new Roman','fontsize',fontsize,'interpreter','latex');
legend('Maximum jitter range', 'Mean jitter range', 'End-effector position', 'FontName', 'times new Roman','fontsize',fontsize)

max(error1);
max(error2);
max(error3);

subplot(2,2,1);
plot(t,theta(1,:)/255*0.5/2, '--',  'linewidth', 3); hold on;
plot(t,theta(2,:)/255*0.5/2,':',  'linewidth', 3);
plot(t,theta(3,:)/255*0.5/2, '-',  'linewidth', 3);
plot(t,theta(4,:)/255*0.5/2,'--',  'linewidth', 2);
plot(t,theta(5,:)/255*0.5/2,':',  'linewidth', 2);
plot(t,theta(6,:)/255*0.5/2,'-',  'linewidth', 2);
set(gca,'FontSize',fontsize, 'FontName', 'times new Roman');
yMin = 0;
yMax = 0.12;
xMin = 0;
xMax = 90;
axis([xMin xMax yMin yMax]);
set(gca, 'xtick', xMin:30:xMax);
set(gca, 'ytick', yMin:0.02:yMax);
xlabel('Time (s)', 'FontName', 'times new Roman','fontsize',fontsize );
ylabel('Muscle pressure (MPa)', 'FontName', 'times new Roman','fontsize',fontsize);
legend('Muscle 1', 'Muscle 2', 'Muscle 3', 'Muscle 4', 'Muscle 5', 'Muscle 6', 'FontName', 'times new Roman','fontsize',fontsize,'numColumns',3)

Error = [y1; y2; y3] - xa;
Mean_Jitter = round(sqrt(sum(Error(1,:).^2 + Error(2,:).^2 + Error(3,:).^2)/number), 4);
Max_Jitter=max(sqrt(Error(1,:).^2 + Error(2,:).^2 + Error(3,:).^2));
