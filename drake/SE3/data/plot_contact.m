%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Contact Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Read the data from the CSV file into a table
filename = 'data/data_contacts.csv';  % Replace with your actual filename
opts = detectImportOptions(filename, 'Delimiter', ',');
opts = setvartype(opts, {'ContactPoint', 'NormalForce'}, 'char'); % Treat arrays as character arrays
dataTable = readtable(filename, opts);

% Process each row
numRows = height(dataTable);
time = zeros(numRows, 1);
bodyA = cell(numRows, 1);
bodyB = cell(numRows, 1);
contactPoint = zeros(numRows, 3);
normalForce = zeros(numRows, 3);

for i = 1:numRows
    % Extract values from the table
    time(i) = dataTable.Time(i);
    bodyA{i} = dataTable.BodyA{i};
    bodyB{i} = dataTable.BodyB{i};
    
    % Parse contact point and normal force arrays
    contactPointStr = dataTable.('ContactPoint'){i};
    normalForceStr = dataTable.('NormalForce'){i};
    
    % Remove brackets and convert to numeric
    contactPoint(i, :) = str2num(strrep(strrep(contactPointStr, '[', ''), ']', '')); %#ok<ST2NM>
    normalForce(i, :) = str2num(strrep(strrep(normalForceStr, '[', ''), ']', '')); %#ok<ST2NM>
end

% Display the data
for i = 1:numRows
    fprintf('Time: %.3f\n', time(i));
    fprintf('Body A: %s\n', bodyA{i});
    fprintf('Body B: %s\n', bodyB{i});
    fprintf('Contact Point: [%.8f %.8f %.8f]\n', contactPoint(i, :));
    fprintf('Normal Force: [%.8f %.8f %.8f]\n\n', normalForce(i, :));
end

