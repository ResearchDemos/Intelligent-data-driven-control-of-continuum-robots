
clear all
load data/val_results.mat
fontsize=11;
subplot( dim , 1 , 1 );
ylabel( 'X' , 'FontSize', fontsize, 'interpreter','latex', 'Rotation', pi/2 );
ylim([-1,1]);
hold on;
plot(  t , dx_a( : , 1 ) , '-', 'linewidth',2);
plot(  t , dx_p( : , 1 ) , '--','linewidth',2 );
set(gca, 'FontSize', fontsize)
hold off;

subplot( dim , 1 , 2 );
ylabel( 'Y' , 'FontSize', fontsize, 'interpreter','latex' , 'Rotation', pi/2);
ylim([-1,1]);
hold on;
plot(  t , dx_a( : , 2 ) , '-', 'linewidth',2);
plot(  t , dx_p( : , 2 ) , '--','linewidth',2 );
set(gca, 'FontSize', fontsize)
hold off;

subplot( dim, 1 , 3 );
ylabel( 'Z' , 'FontSize', fontsize, 'interpreter','latex', 'Rotation', pi/2 );
ylim([-1,1]);
hold on;
plot( t , dx_a( : , 3 ) , '-', 'linewidth',2);
plot( t , dx_p( : , 3 ) , '--','linewidth',2 );
set(gca, 'FontSize', fontsize)
hold off;


legend({'Real value' , 'Prediction'}, 'FontSize', fontsize, 'NumColumns', 2);