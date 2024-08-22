%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Simulation Data, 3D with no arms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load the csv data
data = csvread('data_SE3.csv');

% Extract the data
nq = 17;
nv = 16;
t_data = data(:,1);
q_data = data(:,2:18);
v_data = data(:,19:34);

% plot only a desired segments of the data
t0 = 0;
tf = t_data(end);
idx = find(t_data >= t0 & t_data <= tf);
t_data = t_data(idx);
q_data = q_data(idx,:);
v_data = v_data(idx,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % labels
% q_labels = ["q_{w}", "q_{x}", "q_{y}", "q_{z}", ...
%             "p_x", "p_y", "p_z", ...
%             "q_{HLY}", "q_{HLR}", "q_{HLP}", "q_{KLP}", "q_{FLP}", ...
%             "q_{HRY}", "q_{HRR}", "q_{HRP}", "q_{KRP}", "q_{FRP}"];
% v_labels = ["\omega_x", "\omega_y", "\omega_y",...
%             "v_x", "v_y", "v_z", ...
%             "\dot{q}_{HLY}", "\dot{q}_{HLR}", "\dot{q}_{HLP}", "\dot{q}_{KLP}", "\dot{q}_{FLP}", ...
%             "\dot{q}_{HRY}", "\dot{q}_{HRR}", "\dot{q}_{HRP}", "\dot{q}_{KRP}", "\dot{q}_{FRP}"];
% q_labels = strcat("$", q_labels, "$");
% v_labels = strcat("$", v_labels, "$");

% joint_labels = {'L Hip Yaw', 'L Hip Roll', 'L Hip Pitch', 'L Knee', 'L Foot', ...
%                 'R Hip Yaw', 'R Hip Roll', 'R Hip Pitch', 'R Knee', 'R Foot'};
% joint_titles = {'Left Hip Yaw', 'Left Hip Roll', 'Left Hip Pitch', 'Left Knee Pitch', 'Left Foot Pitch', ...
%                 'Right Hip Yaw', 'Right Hip Roll', 'Right Hip Pitch', 'Right Knee Pitch', 'Right Foot Pitch'};

% % plot the states
% figure('Name', 'State Data');
% tabgp = uitabgroup;

% % plot positions
% tab = uitab(tabgp, 'Title', 'Orientation');
% axes('Parent', tab);
% for i = 1:nq
%     subplot(3,6,i);
%     plot(t_data, q_data(:,i),'b','LineWidth', 1.5);
%     title(q_labels(i), 'interpreter', 'latex');
%     hold on; grid on;
% end

% % plot velocities
% tab = uitab(tabgp, 'Title', 'Velocity');
% axes('Parent', tab);
% for i = 1:nv
%     subplot(4,4,i);
%     plot(t_data, v_data(:,i),'r','LineWidth', 1.5);
%     title(v_labels(i), 'interpreter', 'latex');
%     hold on; grid on;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot individual phase plots
figure('Name', 'Phase Plots');
joint_idx = 9;  % Left Hip Yaw (8), Left Hip Roll (9), Left Hip Pitch (10), Left Knee Pitch (11), Left Foot Pitch (12)
                % Right Hip Yaw (13), Right Hip Roll (14), Right Hip Pitch (15), Right Knee Pitch (16), Right Foot Pitch (17)
q_data = q_data(:, joint_idx);
v_data = v_data(:, joint_idx-1);

% make a movie
xlims = [min(q_data) - 0.1, max(q_data) + 0.1];
ylims = [min(v_data) - 0.1, max(v_data) + 0.1];
xlim(xlims); ylim(ylims);  
yline(0);
grid on; hold on;
plot(nan, nan);

% animate the data
tic
t_end = t_data(end);
idx = 2;

% num_points_to_keep = length(t_data);
num_points_to_keep = 50;
line_objects = []; % Initialize an empty list to store line objects

while idx <= length(t_data)

    % super title
    msg = sprintf('Time: %.2f s', t_data(idx));
    sgtitle(msg);

    % plot the data now
    line = plot([q_data(idx-1) q_data(idx)], [v_data(idx-1) v_data(idx)], 'b-', 'LineWidth', 1.5);
    dot = plot(q_data(idx), v_data(idx), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    xlabel("$q$", 'interpreter', 'latex');
    ylabel("$\dot{q}$", 'interpreter', 'latex');

    % Add the new line object to the list
    line_objects = [line_objects, line];

    % If there are more than num_points_to_keep line objects, delete the oldest one
    if length(line_objects) > num_points_to_keep
        delete(line_objects(1));
        line_objects(1) = []; % Remove the oldest line object from the list
    end

    % Define a colormap that transitions from blue to white
    color_map = [linspace(1, 0, num_points_to_keep)', linspace(1, 0, num_points_to_keep)', ones(num_points_to_keep, 1)];

    % Update the color of each line in the list
    for i = 1:length(line_objects)
        set(line_objects(i), 'Color', color_map(i, :));
    end

    % draw the plot
    drawnow;

    while toc < t_data(idx)
        % wait until the next time step
    end

    % remove the dot
    delete(dot);

    % increment the index
    idx = idx + 1;
end
