%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Simulation Data, 3D with no arms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load the csv data
controller = 'MH';
time_data = csvread(strcat('data_times_', controller, '.csv'));
state_data = csvread(strcat('data_states_', controller, '.csv'));
torque_data = csvread(strcat('data_torques_', controller, '.csv'));
disturbance_data = csvread(strcat('data_disturbances_', controller, '.csv'));
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
t0 = 0;
tf = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STATE and INPUT
plot_state = 0;
plot_torque = 0;

% COMMANDS
plot_joy = 0;

% PHASE
plot_phase = 0;
plot_phase_movie = 1;
save_phase_movie = 1;

% DISTURBANCE
plot_disturbance = 0;

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
disturbance_data = disturbance_data(idx,:);
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

% extract teh disturbance data
F = disturbance_data(:, 4:6);  % forces in world frame
M = disturbance_data(:, 1:3);  % moments in world frame

if plot_disturbance == 1
    % disturbance labels
    disturbance_F_labels = ["f_{x}", "f_{y}", "f_{z}"];
    disturbance_M_labels = ["M_{x}", "M_{y}", "M_{z}"];
    disturbance_F_labels = strcat("$", disturbance_F_labels, "$");
    disturbance_M_labels = strcat("$", disturbance_M_labels, "$");

    % plot the disturbance data
    figure('Name', 'Disturbance Data');
    tabgp = uitabgroup;

    % plot the forces
    tab = uitab(tabgp, 'Title', 'Forces');
    axes('Parent', tab);
    for i = 1:size(F,2)
        subplot(3,1,i);
        plot(t_data, F(:,i),'r','LineWidth', 1.5);
        xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(disturbance_F_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        grid on;
    end

    % plot the moments
    tab = uitab(tabgp, 'Title', 'Moments');
    axes('Parent', tab);
    for i = 1:size(M,2)
        subplot(3,1,i);
        plot(t_data, M(:,i),'r','LineWidth', 1.5);
        xlabel('Time [s]', 'FontSize', 14, 'Interpreter', 'latex');
        ylabel(disturbance_M_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
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

    leg_idx = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14];
    arm_idx = [6, 7, 8, 9, 15, 16, 17, 18];

    q_leg_data = q_joint_data(:,leg_idx);
    v_leg_data = v_joint_data(:,leg_idx);
    q_arm_data = q_joint_data(:,arm_idx);
    v_arm_data = v_joint_data(:,arm_idx);

    q_leg_labels = q_joint_labels(leg_idx);
    q_leg_labels = strcat("$", q_leg_labels, "$");
    v_leg_labels = v_joint_labels(leg_idx);
    v_leg_labels = strcat("$", v_leg_labels, "$");
    q_arm_labels = q_joint_labels(arm_idx);   
    q_arm_labels = strcat("$", q_arm_labels, "$");
    v_arm_labels = v_joint_labels(arm_idx);
    v_arm_labels = strcat("$", v_arm_labels, "$");

    figure('Name', 'Phase Plot');
    tabgp = uitabgroup;

    % plot the leg phase plots
    tab = uitab(tabgp, 'Title', 'Leg Phase');
    axes('Parent', tab);
    for i = 1:size(q_leg_data,2)
        subplot(2,5,i)
        plot(q_leg_data(:,i), v_leg_data(:,i), 'b', 'LineWidth', 1.5);
        xlabel(q_leg_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        ylabel(v_leg_labels(i), 'FontSize', 16, 'Interpreter', 'latex');
        grid on;
    end

    % plot the arm phase plots
    tab = uitab(tabgp, 'Title', 'Arm Phase');
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

    figure('Name', 'Phase Plot', 'Position', [0, 0, 560, 420]);

    joint_idx = [11];  % Joint indices
    q_joint = q_data(:, joint_idx);
    v_joint = v_data(:, joint_idx - 1);
    
    % Set axis limits
    xlims = [min(q_joint) - 0.05, max(q_joint) + 0.05];
    ylims = [min(v_joint) - 0.05, max(v_joint) + 0.05];
    xlim(xlims);
    ylim(ylims);
    yline(0);
    grid on;
    hold on;
    
    % Customize tick size and font size
    ax = gca;  % Get current axes
    ax.FontSize = 14;  % Set font size for axis labels and ticks
    ax.TickLength = [0.02, 0.02];  % Increase the tick length
    
    % Down-sample the data for real-time plotting
    hz_data = 1 / mean(diff(t_data));
    hz_des = 350;
    down_sample_factor = round(hz_data / hz_des);
    
    t_data = t_data(1:down_sample_factor:end);
    q_joint = q_joint(1:down_sample_factor:end);
    v_joint = v_joint(1:down_sample_factor:end);
    
    % Define how long points stay visible in the plot
    hold_sec = 0.35;  % Duration in seconds that each point stays visible
    
    % Initialize video writer if saving the movie
    if save_phase_movie == 1
        video_filename = 'phase_plot.avi';
        video = VideoWriter(video_filename);
        open(video);
    end
    
    % Determine the number of points to keep visible
    num_points_to_keep = round(hz_des * hold_sec);
    line_objects = [];  % Initialize an empty list to store line handles
    
    % Animation loop setup
    idx = 2;  % Ensure idx starts from 2
    startTime = tic;
    
    while idx <= length(t_data)
        % Create a new line segment and store its handle
        new_line = plot([q_joint(idx-1), q_joint(idx)], [v_joint(idx-1), v_joint(idx)], ...
                        'LineWidth', 3.5, 'Color', [1, 1, 1]);  % Start with white color
        
        % Add the new line object to the list
        line_objects = [line_objects, new_line];
        
        % Limit the number of line objects to num_points_to_keep
        if length(line_objects) > num_points_to_keep
            % Remove the oldest line object
            delete(line_objects(1));
            line_objects(1) = [];  % Remove from the list
        end
        
        % Update the colormap for the reversed heat map effect (from blue to white)
        num_existing_points = length(line_objects);
        for i = 1:num_existing_points
            % Reverse the color transition: white (newest) to blue (oldest)
            fade_factor = 1 - (i - 1) / max(1, (num_existing_points - 1));  % Reverse scaling: 1 (newest) to 0 (oldest)
            fade_factor = min(max(fade_factor, 0), 1);  % Clamp fade_factor to [0, 1]
            set(line_objects(i), 'Color', [fade_factor, fade_factor, 1]);  % Transition from blue to white
        end
        
        % Update dot position
        dot = plot(q_joint(idx), v_joint(idx), 'ro', 'MarkerSize', 7, 'MarkerFaceColor', 'r');
    
        % Update plot title and set title size
        titleStr = sprintf('Time: %.2f s', t_data(idx));
        sgtitle(titleStr, 'FontSize', 14);  % Set title size
            
        % Efficient draw call
        drawnow limitrate;
        
        % Capture the current frame if saving a movie
        if save_phase_movie == 1
            frame = getframe(gcf);
            % Resize frame to match figure dimensions if needed
            if size(frame.cdata, 1) ~= 420 || size(frame.cdata, 2) ~= 560
                resized_frame = imresize(frame.cdata, [420, 560]);
                writeVideo(video, resized_frame);
            else
                writeVideo(video, frame);
            end
        end
        
        % Clear dot position for the next frame
        set(dot, 'XData', nan, 'YData', nan);
        
        % Increment index
        idx = idx + 1;
    end
    
    % Close video if recording
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



function replay_phase_plot(replay_data, t_data)
    % Function to replay the phase plot
    figure('Name', 'Replay Phase Plot', 'Position', [100, 100, 560, 420]);
    hold on;
    xlim([min(replay_data(:, 1)) - 0.05, max(replay_data(:, 1)) + 0.05]);
    ylim([min(replay_data(:, 2)) - 0.05, max(replay_data(:, 2)) + 0.05]);
    yline(0);
    grid on;

    % Loop through replay data
    for i = 1:size(replay_data, 1) / 2
        plot(replay_data(2*i-1:2*i, 1), replay_data(2*i-1:2*i, 2), 'LineWidth', 1.5, 'Color', [1, 1, 1]);
        pause(0.1);  % Adjust the pause duration for desired replay speed
    end

    % Final title
    sgtitle('Replay of Phase Plot');
end