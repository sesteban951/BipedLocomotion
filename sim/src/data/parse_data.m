%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARSE SIM DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Location of main data folder
data_folder = "./data/";

% load the data
time = load(data_folder + 'times.csv');
state = load(data_folder + 'full_state.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TIME WINDOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% segemnt the data to a time window
t_interval = [time(1), time(end)];
% t_interval = [0, 10.0];
time_idx = find(time >= t_interval(1) & time <= t_interval(2));

time = time(time_idx);
state = state(time_idx, :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STATE INDECES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% orientation in world frame
QUAT_W = 1; % quaternion w
QUAT_X = 2; % quaternion x
QUAT_Y = 3; % quaternion y
QUAT_Z = 4; % quaternion z

% position in world frame
POS_X = 5;  % position x
POS_Y = 6;  % position y
POS_Z = 7;  % position z

% leg jiont positions relative to zero
POS_LHP = 8; % left hip position
POS_LHR = 9; % left hip roll position
POS_LHY = 10; % left hip yaw position
POS_LKP = 11; % left knee position
POS_LAP = 12; % left ankle position
POS_LAR = 13; % left ankle roll position

% right leg joint positions relative to zero
POS_RHP = 14; % right hip position
POS_RHR = 15; % right hip roll position
POS_RHY = 16; % right hip yaw position
POS_RKP = 17; % right knee position
POS_RAP = 18; % right ankle position
POS_RAR = 19; % right ankle roll position

% base velocity in world frame
ANG_X = 20;  % velocity x
ANG_Y = 21;  % velocity y
ANG_Z = 22;  % velocity z

% angular velocity in world frame
VEL_X = 23; % angular velocity x
VEL_Y = 24; % angular velocity y
VEL_Z = 25; % angular velocity z

% left leg joint velocities
VEL_LHP = 26; % left hip velocity
VEL_LHR = 27; % left hip roll velocity
VEL_LHY = 28; % left hip yaw velocity
VEL_LKP = 29; % left knee velocity
VEL_LAP = 30; % left ankle position
VEL_LAR = 31; % left ankle roll velocity

% right leg joint velocities
VEL_RHP = 32; % right hip velocity
VEL_RHR = 33; % right hip roll velocity
VEL_RHY = 34; % right hip yaw velocity
VEL_RKP = 35; % right knee velocity
VEL_RAP = 36; % right ankle position
VEL_RAR = 37; % right ankle roll velocity

% position indeces
idx_base_pos_w = [POS_X, POS_Y, POS_Z];
idx_base_quat_w = [QUAT_W, QUAT_X, QUAT_Y, QUAT_Z];
idx_leg_pos = [POS_LHP, POS_LHR, POS_LHY, POS_LKP, POS_LAP, POS_LAR, ...
               POS_RHP, POS_RHR, POS_RHY, POS_RKP, POS_RAP, POS_RAR];

% velocity indeces
idx_base_vel_w = [VEL_X, VEL_Y, VEL_Z];
idx_base_ang_b = [ANG_X, ANG_Y, ANG_Z];
idx_leg_vel = [VEL_LHP, VEL_LHR, VEL_LHY, VEL_LKP, VEL_LAP, VEL_LAR, ...
               VEL_RHP, VEL_RHR, VEL_RHY, VEL_RKP, VEL_RAP, VEL_RAR];

% isaac lab joint indeces
idx_leg_pos_isaac = [POS_LHP, POS_RHP, ...
                     POS_LHR, POS_RHR, ...
                     POS_LHY, POS_RHY, ...
                     POS_LKP, POS_RKP, ...
                     POS_LAP, POS_RAP, ...
                     POS_LAR, POS_RAR];

idx_leg_vel_isaac = [VEL_LHP, VEL_RHP, ...
                     VEL_LHR, VEL_RHR, ...
                     VEL_LHY, VEL_RHY, ...
                     VEL_LKP, VEL_RKP, ...
                     VEL_LAP, VEL_RAP, ...
                     VEL_LAR, VEL_RAR];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARSE DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract the data
base_quat_w = state(:, idx_base_quat_w);
base_vel_w = state(:, idx_base_vel_w);
base_ang_b = state(:, idx_base_ang_b);

% compute the gravity orientation in the base frame
gravity_orientation_b = zeros(size(base_quat_w, 1), 3);
for i = 1:size(base_quat_w, 1)
    gravity_orientation_b(i, :) = compute_gravity_orientation(base_quat_w(i, :));
end

% compute the base linear velocity in base frame
base_lin_vel_b = zeros(size(base_vel_w, 1), 3);
for i = 1:size(base_vel_w, 1)

    %  convert quat in world frame to a rotation matrix
    R = quaternion_to_rotation(base_quat_w(i, :));

    % convert the base linear velocity from world frame to base frame
    base_lin_vel_w_t = base_vel_w(i, :)';
    base_lin_vel_b(i, :) = R' * base_lin_vel_w_t;
end

% reorder the joint indeces
joint_pos_isaac = state(:, idx_leg_pos_isaac);
joint_vel_isaac = state(:, idx_leg_vel_isaac);

% save the data into a CSV file
output_folder = "./parsed/";
time_file = output_folder + "time.csv";
base_lin_vel_b_file = output_folder + "base_lin_vel_b.csv";
base_ang_vel_b_file = output_folder + "base_ang_vel_b.csv";
gravity_orientation_b_file = output_folder + "gravity_orientation_b.csv";
joint_pos_isaac_file = output_folder + "joint_pos_isaac.csv";
joint_vel_isaac_file = output_folder + "joint_vel_isaac.csv";

writematrix(time, time_file);
writematrix(base_lin_vel_b, base_lin_vel_b_file);
writematrix(base_ang_b, base_ang_vel_b_file);
writematrix(gravity_orientation_b, gravity_orientation_b_file);
writematrix(joint_pos_isaac, joint_pos_isaac_file);
writematrix(joint_vel_isaac, joint_vel_isaac_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELPER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute gravity orientation in base frame
function g = compute_gravity_orientation(q)

    % unpack the quaternion
    qw = q(1);
    qx = q(2);
    qy = q(3);
    qz = q(4);

    % compute the gravity orientation
    g(1) =  2.0 * (-qz * qx + qw * qy);
    g(2) = -2.0 * ( qz * qy + qw * qx);
    g(3) =  1.0 - 2.0 * (qw * qw + qz * qz);

    % pack into a column vector
    g = [g(1); g(2); g(3)];
end

% convert quaternion to rotation matrix
function R = quaternion_to_rotation(q)
    
    % unpack the quaternion
    q = q / norm(q);
    qw = q(1);
    qx = q(2);
    qy = q(3);
    qz = q(4);

    % convert quaternion to rotation matrix
    quat = quaternion(qw, qx, qy, qz);
    R_frame = rotmat(quat, 'frame');
    % R_point = rotmat(quat, 'point')

    R = R_frame;
end 
