% Load the dataset from an Excel file
a1= 'A1081June.csv';
data = readtable(a1); % Read the data from Excel with the specified options
% disp(data);

% Preprocessing steps: remove all other columns except for predictors 
% Select only the desired columns
data = data(:, {'LocalDate', 'LocalTime', 'TotalCarriagewayFlow'});

% Save the edited data to a new CSV file
newFilename = 'A1081_cleaned.csv'; %new file name
writetable(data, newFilename);

% Read cleaned data: 
aclean= 'A1081_cleaned.csv';
dataclean = readtable(aclean); % Read the data from Excel with the specified options
% disp(dataclean);
% Load the cleaned dataset
% data = readtable('A1081_cleaned.csv');

%dataclean is the name of the cleaned dataset

% % Determine the size of the dataset
% dataSize = size(dataclean, 1);
% 
% % Specify the percentage of data to be used for testing
% testRatio = 0.20;
% testSize = floor(dataSize * testRatio);
% 
% % Shuffle the data
% shuffledData = dataclean(randperm(dataSize), :);
% 
% % Split the data
% trainData = shuffledData(1:end-testSize, :);
% testData = shuffledData(end-testSize+1:end, :);
% 
% % Save the train and test sets to CSV files
% writetable(trainData, 'trainData.csv');
% writetable(testData, 'testData.csv');

% % read train
% train= 'trainData.csv';
% train1 = readtable(train);
% disp(train1);

% % read test
% test= 'testData.csv';
% test1 = readtable(test);
% disp(test1);