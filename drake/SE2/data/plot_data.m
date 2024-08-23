%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Simulation Data, 2D with no arms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Load the csv data
data = csvread('data_SE2.csv');

% Extract the data
t_data = data(:,1);
q_data = data(:,2:10);
v_data = data(:,11:19);

% plot only a desired segments of the data
t0 = 0;
% tf = 5;
tf = t_data(end);
idx = find(t_data >= t0 & t_data <= tf);
t_data = t_data(idx);
q_data = q_data(idx,:);
v_data = v_data(idx,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% labels
titles = ["X Pos", "Z Pos", "Theta Orient", "HipLeftPitch", "KneeLeftPitch", "FootLeftPitch", "HipRightPitch", "KneeRightPitch", "FootRightPitch"];
q_labels = ["x", "z", "\theta", "q_{HLP}", "q_{KLP}", "q_{FLP}", "q_{HRP}", "q_{KRP}", "q_{FRP}"];
v_labels = ["\dot{x}", "\dot{z}", "\dot{\theta}", "\dot{q}_{HLP}", "\dot{q}_{KLP}", "\dot{q}_{FLP}", "\dot{q}_{HRP}", "\dot{q}_{KRP}", "\dot{q}_{FRP}"];
q_labels = strcat("$", q_labels, "$");
v_labels = strcat("$", v_labels, "$");

joint_labels = {"$q_{HLP}$", "$q_{KLP}$", "$q_{FLP}$", "$q_{HRP}$", "$q_{KRP}$", "$q_{FRP}$"};
joint_titles = {"Hip Left Pitch", "Knee Left Pitch", "Foot Left Pitch", "Hip Right Pitch", "Knee Right Pitch", "Foot Right Pitch"};

% % plot the states
% figure('Name', 'State Data');
% tabgp = uitabgroup;
% for i = 1:9
%     tab = uitab(tabgp, 'Title', titles{i});
%     axes('Parent', tab);
    
%     subplot(2,1,1);
%     plot(t_data, q_data(:,i),'b','LineWidth', 1.5);
%     ylabel(q_labels{i}, 'interpreter', 'latex');
%     xlabel('Time (s)');
%     grid on;

%     subplot(2,1,2);
%     plot(t_data, v_data(:,i),'r','LineWidth', 1.5);
%     ylabel(v_labels{i}, 'interpreter', 'latex');
%     xlabel('Time (s)');
%     grid on;

% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % create the figure
% figure('Name', 'Phase Plot Movie');

% % Set the renderer to 'painters'
% set(gcf, 'renderer', 'painters');

% % create the subplots
% for i = 1:6
%     subplot(2, 3, i); 
%     plot(nan, nan);
%     xlims = [min(q_data(:, i+3)) - 0.1, max(q_data(:, i+3)) + 0.1];
%     ylims = [min(v_data(:, i+3)) - 0.1, max(v_data(:, i+3)) + 0.1];
%     plot(nan, nan);
%     xlim(xlims); ylim(ylims);
%     grid on; hold on;
% end

% % animate the data
% tic
% t_end = t_data(end);
% idx = 2;
% while idx <= length(t_data)

%     % super title
%     msg = sprintf('Time: %.2f s', t_data(idx));
%     sgtitle(msg);

%     % Create subplots for each joint
%     for i = 1:6
%         subplot(2, 3, i);
%         line = plot([q_data(idx-1, i+3) q_data(idx, i+3)], [v_data(idx-1, i+3) v_data(idx, i+3)], 'b-', 'LineWidth', 1.5);
%         dot = plot(q_data(idx, i+3), v_data(idx, i+3), 'ro', 'MarkerSize', 2, 'MarkerFaceColor', 'r');
%         xlabel("$q$", 'interpreter', 'latex');
%         ylabel("$\dot{q}$", 'interpreter', 'latex');
%         title(joint_titles{i});
%     end

%     % plot the data now
%     drawnow;

%     while toc < t_data(idx)
%         % wait until the next time step
%     end

%     % remove the dot
%     for i = 1:6
%         subplot(2, 3, i);
%         delete(dot);
%     end

%     % increment the index
%     idx = idx + 1;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot individual phase plots
figure('Name', 'Phase Plots');
joint_idx = 4;  % Left Hip Pitch (3), Left Knee Pitch (4), Left Foot Pitch (5), 
                % Right Hip Pitch (6), Right Knee Pitch (7), Right Foot Pitch (8)
q_data = q_data(:, joint_idx);
v_data = v_data(:, joint_idx);

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
while idx <= length(t_data)

    % super title
    msg = sprintf('Time: %.2f s', t_data(idx));
    sgtitle(msg);

    % plot the data now
    line = plot([q_data(idx-1) q_data(idx)], [v_data(idx-1) v_data(idx)], 'b-', 'LineWidth', 1.5);
    dot = plot(q_data(idx), v_data(idx), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    xlabel("$q$", 'interpreter', 'latex');
    ylabel("$\dot{q}$", 'interpreter', 'latex');

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
