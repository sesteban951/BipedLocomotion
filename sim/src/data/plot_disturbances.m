%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Disturbances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load the csv data
hlip_data = readmatrix('data_random_disturbance_hlip.csv');
% mpc_data = readmatrix('data_random_disturbance_mpc.csv');
mh_data = readmatrix('data_random_disturbance_mh.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% import the yaml file
yaml_file = '../../config/config.yaml';
config = yaml.loadFile(yaml_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parse data that is failed and data that succeeded
hlip_fell = hlip_data(hlip_data(:,4) == 1, :);
hlip_succeeded = hlip_data(hlip_data(:,4) == 0, :);

% mpc_fell = mpc_data(mpc_data(:,4) == 0, :);
% mpc_succeeded = mpc_data(mpc_data(:,4) == 1, :);

mh_fell = mh_data(mh_data(:,4) == 1, :);
mh_succeeded = mh_data(mh_data(:,4) == 0, :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

blue = [0, 0.353, 0.710];
red = [0.980, 0.196, 0.125];
orange = [1, 0.5, 0];
green = [0, 1, 0];
black = [0.2, 0.2, 0.2];

% to plot a cirlce
thetas = 0:0.001:2*pi;
circle_x = cos(thetas);
circle_y = sin(thetas);
r = config.disturbance.force_radius;
xlims = [-r*1.2, r*1.2];
ylims = [-r*1.2, r*1.2];

figure('Name', 'Disturbance Plot');
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');

subplot(1,2,1);
hold on;
% plot(hlip_succeeded(:,2), hlip_succeeded(:,1), 'color', blue, 'Marker', '.', 'MarkerSize', 10, 'LineWidth', 2);
plot(hlip_fell(:,2), hlip_fell(:,1), 'Color', red, 'Marker', 'x',  'MarkerSize', 8,'LineStyle', 'none', 'LineWidth', 2);
plot(hlip_succeeded(:,2), hlip_succeeded(:,1), 'Color', blue, 'Marker', '.', 'MarkerSize', 20, 'LineStyle', 'none', 'LineWidth', 2);
plot(r*circle_x, r*circle_y, 'k--', 'LineWidth', 2);
xlabel('Lateral Disturbance [$N$]', 'interpreter', 'latex');
ylabel('Forward Disturbance [$N$]', 'interpreter', 'latex');
title('HLIP', 'interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12, 'LineWidth', 2);
xlim(xlims);
ylim(ylims);
legend('Fell', 'Succeeded', 'Location', 'best', 'interpreter', 'latex');

subplot(1,2,2);
hold on;
plot(mh_fell(:,2), mh_fell(:,1), 'Color', red, 'Marker', 'x',  'MarkerSize', 8,'LineStyle', 'none', 'LineWidth', 2);
plot(mh_succeeded(:,2), mh_succeeded(:,1), 'Color', blue, 'Marker', '.', 'MarkerSize', 20, 'LineStyle', 'none', 'LineWidth', 2);
plot(r*circle_x, r*circle_y, 'k--', 'LineWidth', 2);
xlabel('Lateral Disturbance [$N$]', 'interpreter', 'latex');
ylabel('Forward Disturbance [$N$]', 'interpreter', 'latex');
title('HLIP + CI-MPC', 'interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12, 'LineWidth', 2);
xlim(xlims);
ylim(ylims);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%