%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Simulation Data, 3D with no arms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load the csv data
controller = 'MH';
time_data_MH = csvread(strcat('data_times_', controller, '.csv'));
state_data_MH = csvread(strcat('data_states_', controller, '.csv'));

controller = 'MPC';
time_data_MPC = csvread(strcat('data_times_', controller, '.csv'));
state_data_MPC = csvread(strcat('data_states_', controller, '.csv'));

controller = 'HLIP';
time_data_HLIP = csvread(strcat('data_times_', controller, '.csv'));
state_data_HLIP = csvread(strcat('data_states_', controller, '.csv'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% import the yaml file
yaml_file = '../../config/config.yaml';
config = yaml.loadFile(yaml_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot only a desired segments of the data
t_data_MH = time_data_MH;
t_data_MPC = time_data_MPC;
t_data_HLIP = time_data_HLIP;

t0 = 0;
tf = 16;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot settings
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract the data
nq = 25;
nv = 24;
q_data_MH = state_data_MH(:,1:nq);
v_data_MH = state_data_MH(:,nq+1:nq+nv);
q_data_MPC = state_data_MPC(:,1:nq);
v_data_MPC = state_data_MPC(:,nq+1:nq+nv);
q_data_HLIP = state_data_HLIP(:,1:nq);
v_data_HLIP = state_data_HLIP(:,nq+1:nq+nv);

% Data extraction with respect to time range
idx = find(t_data_MH >= t0 & t_data_MH <= tf);
t_data_MH = t_data_MH(idx);
q_data_MH = q_data_MH(idx,:);
v_data_MH = v_data_MH(idx,:);

idx = find(t_data_MPC >= t0 & t_data_MPC <= tf);
t_data_MPC = t_data_MPC(idx);
q_data_MPC = q_data_MPC(idx,:);
v_data_MPC = v_data_MPC(idx,:);

idx = find(t_data_HLIP >= t0 & t_data_HLIP <= tf);
t_data_HLIP = t_data_HLIP(idx);
q_data_HLIP = q_data_HLIP(idx,:);
v_data_HLIP = v_data_HLIP(idx,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% state data
q_base_data_MH = q_data_MH(:,1:7);
v_base_data_MH = v_data_MH(:,1:6);

q_base_data_MPC = q_data_MPC(:,1:7);
v_base_data_MPC = v_data_MPC(:,1:6);

q_base_data_HLIP = q_data_HLIP(:,1:7);
v_base_data_HLIP = v_data_HLIP(:,1:6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot the reference
t_ref = t0:0.01:tf;
t_ramp_up = config.references.t_ramp_up;
t_hold = config.references.t_hold;
t_ramp_down = config.references.t_ramp_down;
v_max = config.references.vx_ref; % Assuming v_max is defined in the config

% Initialize the velocity array
v_ref = zeros(size(t_ref));

% Build the reference velocity signal
for i = 1:length(t_ref)
    if t_ref(i) < t_ramp_up
        v_ref(i) = v_max / t_ramp_up * t_ref(i);
    elseif t_ref(i) < t_ramp_up + t_hold
        v_ref(i) = v_max;
    elseif t_ref(i) < t_ramp_up + t_hold + t_ramp_down
        v_ref(i) = v_max - v_max / t_ramp_down * (t_ref(i) - t_ramp_up - t_hold);
    else
        v_ref(i) = 0;
    end
end

% colors
red = [216 27 96]; 
red = red/norm(red);
blue = [30 136 229];
blue = blue/norm(blue);
yellow = [255 193 7];
yellow = yellow/norm(yellow);
green = [0 77 64];
green = green/norm(green);
black = [0 0 0];

% plot the x-velocity
figure;
font_size = 14;
line_width = 2;
tick_size = 16; % Define tick size
hold on; grid on;


plot(t_data_HLIP, v_base_data_HLIP(:,4), 'Color', red, 'LineWidth', line_width); % Red for HLIP
plot(t_data_MPC, v_base_data_MPC(:,4), 'Color', yellow, 'LineWidth', line_width); % Yellow for CI-MPC
plot(t_data_MH, v_base_data_MH(:,4), 'Color', blue, 'LineWidth', line_width); % Blue for Proposed
plot(t_ref, v_ref, 'Color', black, 'LineWidth', line_width, 'LineStyle', '--'); % Green dashed for Reference

xlabel('Time [$s$]', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('$v_x$ [$m/s$]', 'Interpreter', 'latex', 'FontSize', font_size);
legend('HLIP', 'CI-MPC', 'Proposed', 'Reference', 'interpreter', 'latex', 'FontSize', font_size);

set(gca, 'FontSize', tick_size, 'TickLabelInterpreter', 'latex');
% Explicitly set x-axis limit to ensure it ends at 16 seconds
xlim([t0 tf]);