%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Contact Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Read the CSV file into a table
filename = 'data_contacts.csv';
data = readtable(filename);

% import the yaml file
yaml_file = '../../config/config.yaml';
config = yaml.loadFile(yaml_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% choose what to plot
plot_foot_position = 0;
plot_foot_boolean = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot only a desired segments of the data
dt = config.MPC.dt;
t0 = min(data.Time);
tf = max(data.Time);
% t0 = 4;
% tf = 6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract individual columns
time = data.Time;   % Extract time column
BodyA = data.BodyA; % Extract Body A column
BodyB = data.BodyB; % Extract Body B column

% Convert string representations of arrays into actual arrays
pos = cellfun(@(x) str2num(x), data.ContactPoint, 'UniformOutput', false); % Contact point column
force = cellfun(@(x) str2num(x), data.NormalForce, 'UniformOutput', false); % Normal force column

% Find the indices corresponding to the time range
start_idx = find(time >= t0, 1, 'first');
end_idx = find(time <= tf, 1, 'last');

% Extract the relevant segments of the data based on the time range
time = time(start_idx:end_idx);
time_unique = unique(time);
BodyA = BodyA(start_idx:end_idx);
BodyB = BodyB(start_idx:end_idx);
pos = pos(start_idx:end_idx);
force = force(start_idx:end_idx);

% body names
world = "world";
left_foot_heel = "left_foot_heel";
left_foot_toe = "left_foot_toe";
right_foot_heel = "right_foot_heel";
right_foot_toe = "right_foot_toe";

% Find the indices of the rows that contain the body name NOTE: can also AND the two conditions
left_foot_heel_idx = find(contains(BodyA, left_foot_heel) | contains(BodyB, left_foot_heel));
left_foot_toe_idx = find(contains(BodyA, left_foot_toe) | contains(BodyB, left_foot_toe));
right_foot_heel_idx = find(contains(BodyA, right_foot_heel) | contains(BodyB, right_foot_heel));
right_foot_toe_idx = find(contains(BodyA, right_foot_toe) | contains(BodyB, right_foot_toe));

% Extract the data for the selected body
left_foot_heel_time = time(left_foot_heel_idx);
left_foot_toe_time = time(left_foot_toe_idx);
right_foot_heel_time = time(right_foot_heel_idx);
right_foot_toe_time = time(right_foot_toe_idx);

left_foot_heel_pos = cell2mat(pos(left_foot_heel_idx));
left_foot_toe_pos = cell2mat(pos(left_foot_toe_idx));
right_foot_heel_pos = cell2mat(pos(right_foot_heel_idx));
right_foot_toe_pos = cell2mat(pos(right_foot_toe_idx));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot the x-y position of the feet
if plot_foot_position == 1

    figure('Name', 'Foot Position');
    hold on; axis equal; grid on;

    % plot the heel and toe positions
    plot(left_foot_heel_pos(:, 1), left_foot_heel_pos(:, 2), 'bo', 'LineWidth', 2, 'MarkerSize', 10);
    plot(left_foot_toe_pos(:, 1), left_foot_toe_pos(:, 2), 'b*', 'LineWidth', 2, 'MarkerSize', 10);
    
    plot(right_foot_heel_pos(:, 1), right_foot_heel_pos(:, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
    plot(right_foot_toe_pos(:, 1), right_foot_toe_pos(:, 2), 'r*', 'LineWidth', 2, 'MarkerSize', 10);
    
    xlabel('X [m]'); ylabel('Y [m]');
    legend('Left Foot Heel', 'Left Foot Toe', 'Right Foot Heel', 'Right Foot Toe');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if plot_foot_boolean == 1
   
    % create a boolean signal
    left_heel_contact_bool_signal = zeros(length(time_unique), 1);
    right_heel_contact_bool_signal = zeros(length(time_unique), 1);
    left_toe_contact_bool_signal = zeros(length(time_unique), 1);
    right_toe_contact_bool_signal = zeros(length(time_unique), 1);

    % left foot heel
    for i = 1:length(time_unique)
        t = find(left_foot_heel_time == time_unique(i));
        if ~isempty(t)
            left_heel_contact_bool_signal(i) = 1;
        end
    end

    % left foot toe
    for i = 1:length(time_unique)
        t = find(left_foot_toe_time == time_unique(i));
        if ~isempty(t)
            left_toe_contact_bool_signal(i) = 1;
        end
    end

    % right foot heel
    for i = 1:length(time_unique)
        t = find(right_foot_heel_time == time_unique(i));
        if ~isempty(t)
            right_heel_contact_bool_signal(i) = -1;
        end
    end

    % right foot toe
    for i = 1:length(time_unique)
        t = find(right_foot_toe_time == time_unique(i));
        if ~isempty(t)
            right_toe_contact_bool_signal(i) = -1;
        end
    end

    % plot the contact boolean
    figure('Name', 'Contact Boolean');
    hold on; grid on;

    % Plot the left contact boolean signal with color under the curve
    blue = [0, 0, 1];
    red = [1, 0, 0];
    orange = [1, 0.5, 0];
    green = [0, 1, 0];
    subplot(4,1,1)
    area(time_unique, left_heel_contact_bool_signal, 'FaceColor',blue, 'EdgeColor', blue, 'FaceAlpha', 0.5, 'LineWidth', 1);
    subplot(4,1,2)
    area(time_unique, left_toe_contact_bool_signal, 'FaceColor', orange, 'EdgeColor', orange, 'FaceAlpha', 0.5, 'LineWidth', 1);
    subplot(4,1,3)
    area(time_unique, right_heel_contact_bool_signal, 'FaceColor', red, 'EdgeColor', red, 'FaceAlpha', 0.5, 'LineWidth', 1);
    subplot(4,1,4)
    area(time_unique, right_toe_contact_bool_signal, 'FaceColor', green, 'EdgeColor', green, 'FaceAlpha', 0.5, 'LineWidth', 1);

    % Customize the plot
    xlabel('Time [s]');
    ylabel('Contact (1 or 0)');
    title('Left and Right Foot Heel Contact Boolean Signal');
    legend('Left Contact', 'Right Contact');
    grid on;

end