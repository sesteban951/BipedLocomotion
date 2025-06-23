%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Simulation Data, 3D with no arms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;


% Load the csv data
time = csvread('./data/times.csv');
q_joint_des = csvread('./data/q_joint_target.csv');
v_joint_des = csvread('./data/v_joint_target.csv');
tau_ff = csvread('./data/torques_ff.csv');

% joint labels
joint_labels = ["LHP", "LHR", "LHY", "LKP", "LAP", "LAR", ...
                "RHP", "RHR", "RHY", "RKP", "RAP", "RAR"];
num_joints = 12;

% plot the joint angles
figure('Name', 'Joint Angles', 'NumberTitle', 'off');
for i = 1:num_joints
    subplot(3, 4, i);
    plot(time, q_joint_des(:, i), 'LineWidth', 2);
    title(joint_labels(i));
    xlabel('Time (s)');
    ylabel('Joint Angle (rad)');
    grid on;
end

% plot the joint velocities
figure('Name', 'Joint Velocities', 'NumberTitle', 'off');
for i = 1:num_joints
    subplot(3, 4, i);
    plot(time, v_joint_des(:, i), 'LineWidth', 2);
    title(joint_labels(i));
    xlabel('Time (s)');
    ylabel('Joint Velocity (rad/s)');
    grid on;
end

% plot the feed forward torques
figure('Name', 'Feed Forward Torques', 'NumberTitle', 'off');
for i = 1:num_joints
    subplot(3, 4, i);
    plot(time, tau_ff(:, i), 'r', 'LineWidth', 2);
    title(joint_labels(i));
    xlabel('Time (s)');
    ylabel('Feed Forward Torque (Nm)');
    grid on;
end
