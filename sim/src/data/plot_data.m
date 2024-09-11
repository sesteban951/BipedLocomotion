%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Simulation Data, 3D with no arms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load the csv data
controller = 'MH';
time_data = csvread(strcat('data_times_', controller, '.csv'));
state_data = csvread(strcat('data_states_', controller, '.csv'));
torque_data = csvread(strcat('data_torques_', controller, '.csv'));
joystick_data = csvread(strcat('data_joystick_', controller, '.csv'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% import the yaml file
yaml_file = '../../config/config.yaml';
config = yaml.loadFile(yaml_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot only a desired segments of the data
t_data = time_data;
% t0 = t_data(1);
% tf = t_data(end);
t0 = 4;
tf = 6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STATE and INPUT
plot_state = 0;
plot_torque = 0;

% COMMANDS
plot_joy = 0;

% PHASE
plot_phase = 1;
plot_phase_movie = 0;
save_phase_movie = 0;

% EFFICIENCY
plot_efficiency = 0;
plot_ref_tracking = 0;

% plot settings
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract the data
nq = 25;
nv = 24;
q_data = state_data(:,1:nq);
v_data = state_data(:,nq+1:nq+nv);
tau_data = torque_data;

% data extraction with respect to time range
idx = find(t_data >= t0 & t_data <= tf);
t_data = t_data(idx);
q_data = q_data(idx,:);
v_data = v_data(idx,:);
tau_data = tau_data(idx,:);
joystick_data = joystick_data(idx,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% state data
q_base_data = q_data(:,1:7);
q_joint_data = q_data(:,8:end);
v_base_data = v_data(:,1:6);
v_joint_data = v_data(:,7:end);

% plot y labels
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

% extract the data
leg_idx = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14];
arm_idx = [6, 7, 8, 9, 15, 16, 17, 18];

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

    vx_max = config.HLIP.vx_max;
    vy_max = config.HLIP.vy_max;
    wz_max = (config.HLIP.wz_max) * (pi/180);
    z_upper = config.HLIP.z_com_upper;
    z_lower = config.HLIP.z_com_lower;

    vx_command = joystick_data(:,2) * vx_max;
    vy_command = joystick_data(:,1) * vy_max;
    wz_command = joystick_data(:,3) * wz_max;
    z_command = joystick_data(:,5) * (z_lower - z_upper) + z_upper;

    commands = [vx_command, vy_command, wz_command, z_command]

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

% plot all phase plots
if plot_phase == 1

    % orbit tail length
    hold_sec = 0.3;

    figure('Name', 'Phase Plot');
    tabgp = uitabgroup;

    q_leg_data = q_joint_data(:,[1,2,3,4,5,10,11,12,13,14]);
    v_leg_data = v_joint_data(:,[1,2,3,4,5,10,11,12,13,14]);
    q_arm_data = q_joint_data(:,[6,7,8,9,15,16,17,18]);
    v_arm_data = v_joint_data(:,[6,7,8,9,15,16,17,18]);

    q_leg_labels = q_joint_labels([1,2,3,4,5,10,11,12,13,14]);
    q_leg_labels = strcat("$", q_leg_labels, "$");
    v_leg_labels = v_joint_labels([1,2,3,4,5,10,11,12,13,14]);
    v_leg_labels = strcat("$", v_leg_labels, "$");
    q_arm_labels = q_joint_labels([6,7,8,9,15,16,17,18]);   
    q_arm_labels = strcat("$", q_arm_labels, "$");
    v_arm_labels = v_joint_labels([6,7,8,9,15,16,17,18]);
    v_arm_labels = strcat("$", v_arm_labels, "$");

    % plot the leg phase plots
    tab = uitab(tabgp, 'Title', 'Leg Torques');
    axes('Parent', tab);
    for i = 1:size(q_leg_data,2)
        subplot(2,5,i)
        plot(q_leg_data(:,i), v_leg_data(:,i), 'b', 'LineWidth', 1.5);
        xlabel(q_leg_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        ylabel(v_leg_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        grid on;
    end

    % plot the arm phase plots
    tab = uitab(tabgp, 'Title', 'Arm Torques');
    axes('Parent', tab);
    for i = 1:size(q_arm_data,2)
        subplot(2,4,i)
        plot(q_arm_data(:,i), v_arm_data(:,i), 'b', 'LineWidth', 1.5);
        xlabel(q_arm_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        ylabel(v_arm_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        grid on;
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot individual phase plots
if plot_phase_movie == 1
    
    figure('Name', 'Phase Plot', 'Position', [100, 100, 560, 420]);
    joint_idx = 11;  % Left Hip Yaw (8), Left Hip Roll (9), Left Hip Pitch (10), Left Knee Pitch (11), Left ankle Pitch (12)
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
    tic = t_data(1);
    t_end = t_data(end);
    idx = 2;

    % num_points_to_keep = length(t_data);
    hold_sec = 0.1;
    num_points_to_keep = round(hz_des * hold_sec);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot the cost of transport
% COT = (tau(t) * omega(t)) / (m * g * v(t))
if plot_efficiency == 1

    % compute the mechanical joint power
    [r, ~] = size(tau_data);
    P = zeros(r, 1);
    for t = 1:r
        % take the dot product of each row of tau with the corresponding row of v
        tau_t = tau_data(t,:);
        v_joint_data_t = v_joint_data(t,:);
        P(t) = abs(tau_t * v_joint_data_t');
    end

    % integrate teh total power to get energy consumed
    total = trapz(t_data, P);

    % plot the mechanical power
    figure('Name', 'State Data');
    tabgp = uitabgroup;
    tab = uitab(tabgp, 'Title', 'Mechanical Power');
    axes('Parent', tab);
    plot(t_data, P, 'k', 'LineWidth', 1.5);
    yline(0);
    xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
    ylabel('Power [W]', 'FontSize', 16, 'Interpreter', 'latex');
    title(['Total Integrated: ', num2str(total)], 'FontSize', 16, 'Interpreter', 'latex');

    % Compute the accumulated energy over time using trapz in a loop
    E = zeros(r, 1);
    for t = 2:r
        E(t) = trapz(t_data(1:t), P(1:t));
    end

    % plot the accumulated energy
    tab = uitab(tabgp, 'Title', 'Accumulated Energy');
    axes('Parent', tab);
    plot(t_data, E, 'g', 'LineWidth', 1.5);
    xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
    ylabel('Energy [J]', 'FontSize', 16, 'Interpreter', 'latex');
    yline(0);

    % compute the COT
    m = 24; % [kg]
    g = 9.81; % [m/s^2]
    COT = zeros(r, 1);
    for t = 1:r
        if abs(v_joint_data(t,1)) < 1e-6
            COT(t) = 0;
        else
            COT(t) = P(t) / (m * g * v_base_data(t,1));
        end
    end

    % plot the COT
    tab = uitab(tabgp, 'Title', 'Cost of Transport');
    axes('Parent', tab);
    plot(t_data, COT, 'm', 'LineWidth', 1.5);
    xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
    ylabel('Cost of Transport [1]', 'FontSize', 16, 'Interpreter', 'latex');
    yline(0);

    % compute the torque squared, ||u||^2 = u1^2 + u2^2 + u3^2  + ...
    [r, ~] = size(tau_data);
    U = zeros(r, 1);
    for t = 1:r
        tau_t = tau_data(t,:);
        U(t) = tau_t * tau_t';
    end
    
    % integrate over the bounds to get the total energy
    total = trapz(t_data, U);

    % plot torque squared
    tab = uitab(tabgp, 'Title', 'Torque Squared');
    axes('Parent', tab);
    plot(t_data, U, 'r', 'LineWidth', 1.5);
    yline(0);
    xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
    ylabel('$\|\tau\|^2_2$, (Nm)$^2$', 'FontSize', 16, 'Interpreter', 'latex');
    title(['Total Integrated: ', num2str(total)], 'FontSize', 16, 'Interpreter', 'latex');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if plot_ref_tracking == 1

    % unpack the reference commands
    vx_ref = config.references.vx_ref;
    vy_ref = config.references.vy_ref;
    wz_ref = config.references.wz_ref;
    z_com_ref = config.references.z_com_ref;

    % plot the reference tracking
    figure('Name', 'Reference Tracking');
    tabgp = uitabgroup;

    % plot the velocity
    tab = uitab(tabgp, 'Title', 'X Velocity');
    axes('Parent', tab);
    hold on; grid on;
    plot(t_data, vx_ref * ones(length(t_data),1), '--k', 'LineWidth', 1.5);
    plot(t_data, v_base_data(:,1), 'b', 'LineWidth', 1.5);

    tab = uitab(tabgp, 'Title', 'Y Velocity');
    axes('Parent', tab);
    hold on; grid on;
    plot(t_data, vy_ref * ones(length(t_data),1), '--k', 'LineWidth', 1.5);
    plot(t_data, v_base_data(:,2), 'b', 'LineWidth', 1.5);

    % plot both x and y velocities
    tab = uitab(tabgp, 'Title', 'X and Y Velocities');
    axes('Parent', tab);
    hold on; grid on;
    plot(v_base_data(:,1), v_base_data(:,2), 'b', 'LineWidth', 1.5);
    plot(vx_ref, vy_ref, 'r+', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);

end