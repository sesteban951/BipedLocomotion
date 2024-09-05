%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Simulation Data, 3D with no arms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load the csv data
time_data = csvread('data_times.csv');
state_data = csvread('data_states.csv');
torque_data = csvread('data_torques.csv');
joystick_data = csvread('data_joystick.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% choose what to plot
plot_state = 0;
plot_torque = 0;
plot_joy = 0;
plot_phase = 1;
plot_phase_movie = 0;
save_phase_movie = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract the data
nq = 25;
nv = 24;
t_data = time_data;
q_data = state_data(:,1:nq);
v_data = state_data(:,nq+1:nq+nv);
tau_data = torque_data;

% plot only a desired segments of the data
t0 = 0;
tf = t_data(end);
idx = find(t_data >= t0 & t_data <= tf);

t_data = t_data(idx);
q_data = q_data(idx,:);
v_data = v_data(idx,:);
tau_data = tau_data(idx,:);

q_base_data = q_data(:,1:7);
q_joint_data = q_data(:,8:end);
v_base_data = v_data(:,1:6);
v_joint_data = v_data(:,7:end);

leg_idx = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14];
arm_idx = [6, 7, 8, 9, 15, 16, 17, 18];

% plot labels
q_base_labels = ["q_{w}", "q_{x}", "q_{y}", "q_{z}", ...
                "p_x", "p_y", "p_z"];
q_joint_labels = ["q_{LHY}", "q_{LHR}", "q_{LHP}", "q_{LKP}", "q_{LAP}", ...
                "q_{LSP}", "q_{LSR}", "q_{LSY}", "q_{LEP}", ...
                "q_{RHY}", "q_{RHR}", "q_{RHP}", "q_{RKP}", "q_{RAP}",...
                "q_{RSP}", "q_{RSR}", "q_{RSY}", "q_{REP}"]; 
v_base_labels = ["\omega_x", "\omega_y", "\omega_y",...
                "v_x", "v_y", "v_z"];
v_joint_labels = ["\dot{q}_{LHY}", "\dot{q}_{LHR}", "\dot{q}_{LHP}", "\dot{q}_{LKP}", "\dot{q}_{LAP}", ...
                "\dot{q}_{LSP}", "\dot{q}_{LSR}", "\dot{q}_{LSY}", "\dot{q}_{LEP}", ...
                "\dot{q}_{RHY}", "\dot{q}_{RHR}", "\dot{q}_{RHP}", "\dot{q}_{RKP}", "\dot{q}_{RAP}",...
                "\dot{q}_{RSP}", "\dot{q}_{RSR}", "\dot{q}_{RSY}", "\dot{q}_{REP}"];
q_base_labels = strcat("$", q_base_labels, "$");
q_joint_labels = strcat("$", q_joint_labels, "$");
v_base_labels = strcat("$", v_base_labels, "$");
v_joint_labels = strcat("$", v_joint_labels, "$");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot the states
if plot_state == 1

    figure('Name', 'State Data');
    tabgp = uitabgroup;

    % plot base positions
    tab = uitab(tabgp, 'Title', 'Base Positions');
    axes('Parent', tab);
    for i = 1:size(q_base_data,2)
        subplot(2,4,i);
        plot(t_data, q_base_data(:,i),'b','LineWidth', 1.5);
        xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(q_base_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        grid on;
    end

    % plot the joint positions
    tab = uitab(tabgp, 'Title', 'Joint Positions');
    axes('Parent', tab);
    for i = 1:size(q_joint_data,2)
        subplot(4,5,i);
        plot(t_data, q_joint_data(:,i),'b','LineWidth', 1.5);
        xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(q_joint_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        grid on;
    end

    % plot base velocities
    tab = uitab(tabgp, 'Title', 'Base Velocity');
    axes('Parent', tab);
    for i = 1:size(v_base_data,2)
        subplot(2,3,i);
        plot(t_data, v_base_data(:,i),'r','LineWidth', 1.5);
        xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(v_base_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        grid on;
    end

    % plot the joint velocities
    tab = uitab(tabgp, 'Title', 'Joint Velocity');
    axes('Parent', tab);
    for i = 1:size(v_joint_data,2)
        subplot(4,5,i);
        plot(t_data, v_joint_data(:,i),'r','LineWidth', 1.5);
        xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(v_joint_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        grid on;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot the torques
if plot_torque == 1

    % torque labels
    tau_joint_labels = ["\tau_{LHY}", "\tau_{LHR}", "\tau_{LHP}", "\tau_{LKP}", "\tau_{LAP}", ...
                    "\tau_{LSP}", "\tau_{LSR}", "\tau_{LSY}", "\tau_{LEP}", ...
                    "\tau_{RHY}", "\tau_{RHR}", "\tau_{RHP}", "\tau_{RKP}", "\tau_{RAP}",...
                    "\tau_{RSP}", "\tau_{RSR}", "\tau_{RSY}", "\tau_{REP}"];
    tau_joint_labels = strcat("$", tau_joint_labels, "$");

    % torque plot
    figure('Name', 'Torque Data');
    tabgp = uitabgroup;

    % leg torques
    tab = uitab(tabgp, 'Title', 'Leg Torques');
    axes('Parent', tab);
    for i = 1:length(leg_idx)
        subplot(2,5,i);
        plot(t_data, tau_data(:,leg_idx(i)),'r','LineWidth', 1.5);
        xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(tau_joint_labels(leg_idx(i)), 'FontSize', 16, 'Interpreter', 'latex');
        grid on;
    
    end

    % arm torques
    tab = uitab(tabgp, 'Title', 'Arm Torques');
    axes('Parent', tab);
    for i = 1:length(arm_idx)
        subplot(2,4,i);
        plot(t_data, tau_data(:,arm_idx(i)),'r','LineWidth', 1.5);
        xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(tau_joint_labels(arm_idx(i)), 'FontSize', 16, 'Interpreter', 'latex');
        grid on;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot the joystick commands
if plot_joy == 1

    vx_max = 0.4;
    vy_max = 0.3;
    wz_max = (25.0) * (pi/180);
    z_upper = 0.65;
    z_lower = 0.45;

    vx_command = joystick_data(:,2) * vx_max;
    vy_command = joystick_data(:,1) * vy_max;
    wz_command = joystick_data(:,3) * wz_max;
    z_command = joystick_data(:,5) * (z_lower - z_upper) + z_upper;

    commands = [vx_command, vy_command, wz_command, z_command];

    command_labels = ["v_x", "v_y", "\omega_z", "z_{com}"];
    command_labels = strcat("$", command_labels, "$");

    figure('Name', 'Joystick Commands');
    for i = 1:size(commands,2)
        subplot(2,2,i);
        plot(t_data, commands(:,i), 'b', 'LineWidth', 1.5);
        xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(command_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        grid on; hold on;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if plot_phase == 1

    % plot all joint phase plots
    figure('Name', 'Phase Plots');
    tabgp = uitabgroup;

    % plot leg phase plots
    tab = uitab(tabgp, 'Title', 'Legs');
    axes('Parent', tab);
    for i = 1:length(leg_idx)
        subplot(2,5,i)
        plot(q_joint_data(:,leg_idx(i)), v_joint_data(:,leg_idx(i)), 'b', 'LineWidth', 1.5);
        xlabel(q_joint_labels(leg_idx(i)), 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(v_joint_labels(leg_idx(i)), 'FontSize', 14, 'Interpreter', 'latex');
        grid on;
    end

    % plot arm phase plots
    tab = uitab(tabgp, 'Title', 'Arms');
    axes('Parent', tab);
    for i = 1:length(arm_idx)
        subplot(2,4,i)
        plot(q_joint_data(:,arm_idx(i)), v_joint_data(:,arm_idx(i)), 'b', 'LineWidth', 1.5);
        xlabel(q_joint_labels(arm_idx(i)), 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(v_joint_labels(arm_idx(i)), 'FontSize', 14, 'Interpreter', 'latex');
        grid on;
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot individual phase plots
if plot_phase_movie == 1
    
    figure('Name', 'Phase Plot', 'Position', [100, 100, 560, 420]);
    joint_idx = 12;  % Left Hip Yaw (8), Left Hip Roll (9), Left Hip Pitch (10), Left Knee Pitch (11), Left ankle Pitch (12)
                    % Left Shoulder Pitch (13), Left Shoulder Roll (14), Left Shoulder Yaw (15), Left Elbow Pitch (16)
                    % Right Hip Yaw (17), Right Hip Roll (18), Right Hip Pitch (19), Right Knee Pitch (20), Right ankle Pitch (21)
                    % Right Shoulder Pitch (22), Right Shoulder Roll (23), Right Shoulder Yaw (24), Right Elbow Pitch (25)
    q_joint = q_data(:, joint_idx);
    v_joint = v_data(:, joint_idx-1);

    % make a movie
    xlims = [min(q_joint) - 0.05, max(q_joint) + 0.05];
    ylims = [min(v_joint) - 0.05, max(v_joint) + 0.05];
    xlim(xlims); ylim(ylims);  
    yline(0);
    grid on; hold on;
    plot(nan, nan);

    % down sample the data to plot in real time
    hz_data = 1 / mean(diff(t_data));
    hz_des = hz_data;
    % hz_des = 30;
    down_sample_factor = round(hz_data / hz_des);

    t_data = t_data(1:down_sample_factor:end);
    q_joint = q_joint(1:down_sample_factor:end);
    v_joint = v_joint(1:down_sample_factor:end);

    % video writer object
    if save_phase_movie == 1
        video_filename = 'phase_plot.avi';
        video = VideoWriter(video_filename);
        open(video);
    end

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
        line = plot([q_joint(idx-1) q_joint(idx)], [v_joint(idx-1) v_joint(idx)], 'b-', 'LineWidth', 1.5);
        dot = plot(q_joint(idx), v_joint(idx), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
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

        % Capture the current frame and write it to the video
        if save_phase_movie == 1
            frame = getframe(gcf);
            % Resize frame to 560x420 if needed
            if size(frame.cdata, 1) ~= 420 || size(frame.cdata, 2) ~= 560
                resized_frame = imresize(frame.cdata, [420, 560]);
                writeVideo(video, resized_frame);
            else
                writeVideo(video, frame);
            end
        end

        while toc < t_data(idx)
            % wait until the next time step
        end

        % remove the dot
        delete(dot);

        % increment the index
        idx = idx + 1;
    end

    % close the video file
    if save_phase_movie == 1
        close(video);
    end
end