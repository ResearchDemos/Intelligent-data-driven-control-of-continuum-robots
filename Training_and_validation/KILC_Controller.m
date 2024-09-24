function [output] = KILC_Controller(t, action_k)
global param data camera kilc_i

idx = round(t/param.Ts)+1;
% tic
markerPose = camera.ReadMarkerPose1(camera);
% toc
tic
data.x_k(:,idx) = markerPose(1:3);
data.action_k(:,idx) = action_k;

y = data.x_k(:,idx);
y_d = data.x_d(:,idx);
dy_d = data.dx_d(:,idx);

nd =  kilc_i.params.nd;
if idx == 1
    ydel = kron( ones( nd , 1 ) ,data.x_k(:,idx));
    udel = kron( ones( nd , 1 ) ,data.u_k(:,idx));
    uidel = kron( ones( nd , 1 ) ,data.action_k(:,idx));
    ydel_d = kron( ones( nd , 1 ) ,data.x_d(:,idx));
    dydel_d = kron( ones( nd , 1 ) ,data.dx_d(:,idx));
    dudel = kron( ones( nd , 1 ) ,data.du_k(:,idx));
elseif idx <  nd+1
    ydel = [get_delay_vector(data.x_k(:,1:idx-1)); kron( ones( nd+1-idx , 1 ) ,data.x_k(:,1))];
    uidel = [get_delay_vector(data.action_k(:,1:idx-1)); kron( ones( nd+1-idx , 1 ) ,data.action_k(:,1))];
    udel = [get_delay_vector(data.u_k(:,1:idx-1)); kron( ones( nd+1-idx , 1 ) ,data.u_k(:,1))];
    ydel_d = [get_delay_vector(data.x_d(:,1:idx-1)); kron( ones( nd+1-idx , 1 ) ,data.x_d(:,1))];
    dydel_d = [get_delay_vector(data.dx_d(:,1:idx-1)); kron( ones( nd+1-idx , 1 ) ,data.dx_d(:,1))];
    dudel = [get_delay_vector(data.du_k(:,1:idx-1)); kron( ones( nd+1-idx , 1 ) ,data.du_k(:,1))];
else 
    ydel = get_delay_vector(data.x_k(:,idx-nd:idx-1));
    uidel = get_delay_vector(data.action_k(:,idx-nd:idx-1));
    udel = get_delay_vector(data.u_k(:,idx-nd:idx-1));
    ydel_d = get_delay_vector(data.x_d(:,idx-nd:idx-1));
    dydel_d = get_delay_vector(data.dx_d(:,idx-nd:idx-1));
    dudel = get_delay_vector(data.du_k(:,idx-nd:idx-1));
end
if kilc_i.params.include_input_delay
    if kilc_i.params.velocity_input_delay
        zeta_k = kilc_i.scaledown.zeta( [ y; ydel; udel ]' )';
        zeta_d = kilc_i.scaledown.zeta( [ y_d; ydel_d; udel ]' )';
        dzeta_d = kilc_i.scaledown.dzeta( [dy_d; dydel_d; dudel*0]' )';
    else
        zeta_k = kilc_i.scaledown.zeta( [ y; ydel; uidel ]' )';
        zeta_d = kilc_i.scaledown.zeta( [ y_d; ydel_d; uidel ]' )';
        dzeta_d = kilc_i.scaledown.dzeta( [dy_d; dydel_d; udel*0]' )';
    end
else
    zeta_k = kilc_i.scaledown.zeta( [ y; ydel ]' )';
    zeta_d = kilc_i.scaledown.zeta( [ y_d; ydel_d ]' )';
    dzeta_d = kilc_i.scaledown.dzeta( [dy_d; dydel_d]' )';
end

phix_k = kilc_i.model.W2_W1 * zeta_k + kilc_i.model.W2_W1_bias + kilc_i.model.W2_bias;
z_k = [zeta_k; RELU(phix_k)];
d_relu = 0*phix_k;
d_relu(phix_k>=0) = 1;
Jacob = [eye(kilc_i.params.nzeta); diag(d_relu)*kilc_i.model.W2_W1];

data.z_k(:,idx) = z_k;

phix_d = kilc_i.model.W2_W1 * zeta_d + kilc_i.model.W2_W1_bias + kilc_i.model.W2_bias;
z_d = [zeta_d; RELU(phix_d)];
dz_d = Jacob * dzeta_d;
data.z_d(:,idx) = z_d;

error_k = z_k - z_d;
% error_k = data.error(:,idx);

K =10*eye(size(error_k, 1));
xi = 0.1;
Gamma = 2;
gamma_1 = 0.01;
gamma_2 = 0.01;
gamma_3 = 0.01;

hat_lambda_km1 = param.hat_lambda_km1(idx);
hat_eta_km1 = param.hat_eta_km1(idx);
hat_zeta_km1 = param.hat_zeta_km1(idx);

% A_k = (kilc_i.model.A-eye(size(kilc_i.model.A,1)))/param.Ts;  
% B_k = kilc_i.model.B/param.Ts;   
A_k = kilc_i.model.A;
B_k = kilc_i.model.B;

KAd = K*error_k + A_k * z_k - dz_d + kilc_i.model.A_bias + kilc_i.model.B_bias;
KAd_norm_norm = norm(KAd) * norm(error_k);
eB_norm = norm(error_k'*B_k);
Be = B_k'* error_k;

hat_lambda_k = hat_lambda_km1 + gamma_1 * KAd_norm_norm;
hat_eta_k = hat_eta_km1 + gamma_2 * eB_norm;
hat_zeta_k = hat_zeta_km1 + gamma_3 * error_k' * sign(error_k);

param.hat_lambda_km1(idx) = hat_lambda_k;
param.hat_eta_km1(idx) = hat_eta_k;
param.hat_zeta_km1(idx) = hat_zeta_k;

rho = max(norm(error_k'*B_k,2)^2, 1e-6);

u_k1 = B_k'/norm(B_k*B_k',2) * (-KAd );
u_k2 = - hat_lambda_k * Be / ((1-xi)*rho) * KAd_norm_norm;
u_k3 = - Be / ((1-xi)*rho) * hat_eta_k * eB_norm;
u_k4 = - Be / ((1-xi)*rho) * hat_zeta_k * error_k' * sign(error_k);

u_k_sc = u_k1 + u_k2 + u_k3 + u_k4;
u_k = kilc_i.scaleup.u(u_k_sc')';
u_k = Limit(u_k,param.limit);
data.u_k(:,idx) = u_k;
if idx > 1
    data.du_k(:,idx) = (data.u_k(:,idx) - data.u_k(:,idx-1)) / param.Ts;
end
output = u_k;

cost=toc;
param.time_cost = param.time_cost+cost;
end