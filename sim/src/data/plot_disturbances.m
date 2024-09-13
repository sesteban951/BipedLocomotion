%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Disturbances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load the csv data
hlip_data = readmatrix('data_random_disturbance_hlip.csv');
mpc_data = readmatrix('data_random_disturbance_mpc.csv');
mh_data = readmatrix('data_random_disturbance_mh.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% import the yaml file
yaml_file = '../../config/config.yaml';
config = yaml.loadFile(yaml_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parse data that is failed and data that succeeded
hlip_fell = hlip_data(hlip_data(:,4) == 0, :);
hlip_succeeded = hlip_data(hlip_data(:,4) == 1, :);

mpc_fell = mpc_data(mpc_data(:,4) == 0, :);
mpc_succeeded = mpc_data(mpc_data(:,4) == 1, :);

mh_fell = mh_data(mh_data(:,4) == 0, :);
mh_succeeded = mh_data(mh_data(:,4) == 1, :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% to plot a cirlce
thetas = 0:0.001:2*pi;
circle_x = cos(thetas);
circle_y = sin(thetas);
r = config.disturbance.force_radius;
xlims = [-r, r];
ylims = [-r, r];

figure('Name', 'Disturbance Plot');
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');

subplot(1,3,1);
hold on; 
plot(hlip_succeeded(:,2), hlip_succeeded(:,1), 'b.', 'MarkerSize', 10, 'LineWidth', 2);
plot(hlip_fell(:,2), hlip_fell(:,1), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(r*circle_x, r*circle_y, 'k--', 'LineWidth', 2);
xlabel('Sideways Disturbance [$N$]', 'interpreter', 'latex');
ylabel('Forward Disturbance [$N$]', 'interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12, 'LineWidth', 2);

subplot(1,3,2);
hold on; 
plot(mpc_succeeded(:,2), mpc_succeeded(:,1), 'b.', 'MarkerSize', 10, 'LineWidth', 2);
plot(mpc_fell(:,2), mpc_fell(:,1), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(r*circle_x, r*circle_y, 'k--', 'LineWidth', 2);
xlabel('Sideways Disturbance [$N$]', 'interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12, 'LineWidth', 2);

subplot(1,3,3);
hold on; 
plot(mh_succeeded(:,2), mh_succeeded(:,1), 'b.', 'MarkerSize', 10, 'LineWidth', 2);
plot(mh_fell(:,2), mh_fell(:,1), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(r*circle_x, r*circle_y, 'k--', 'LineWidth', 2);
xlabel('Sideways Disturbance [$N$]', 'interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12, 'LineWidth', 2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%