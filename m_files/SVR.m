%load
trainData = readtable('trainData.csv');
testData = readtable('testData.csv');

% extract hours 
trainData.Local_Time_Numeric = hours(duration(trainData.('LocalTime')));
testData.Local_Time_Numeric = hours(duration(testData.('LocalTime')));

%%%%%%%%Now, trainData.Hour and testData.Hour contain the hours extracted from LocalTime
% Prepare datasets for SVR
XTrain = trainData.Local_Time_Numeric;
YTrain = trainData.TotalCarriagewayFlow;
XTest = testData.Local_Time_Numeric;
YTest = testData.TotalCarriagewayFlow;

% column vectors
XTrain = XTrain(:);
XTest = XTest(:);

%%%%%%%
% we store all the three predictions into these and create a unified graph:
YTest = testData.TotalCarriagewayFlow;  % Actual data

% Gaussian SVM model with optimized hyperparameters
MdlOptimized_gaussian = fitrsvm(XTrain, YTrain, ...
    'KernelFunction', 'gaussian', ...
    'KernelScale', 0.027, ...
    'BoxConstraint', 50.400);

% poly SVM model with optimized hyperparameters
MdlOptimized_poly = fitrsvm(XTrain, YTrain, ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'BoxConstraint', 10);

% rbf SVM model with optimized hyperparameters
MdlOptimized_rbf = fitrsvm(XTrain, YTrain, ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 2.4409, ...
    'BoxConstraint', 11.9989);

% linear SVM model with optimized hyperparameters
MdlOptimized_linear = fitrsvm(XTrain, YTrain, ...
    'KernelFunction', 'linear', ...
    'BoxConstraint', 10);

YPred_gaussian = predict(MdlOptimized_gaussian, XTest);  % Gaussian predictions
YPred_rbf = predict(MdlOptimized_rbf, XTest);            % RBF predictions
YPred_poly = predict(MdlOptimized_poly, XTest);          % Polynomial predictions
YPred_linear = predict(MdlOptimized_linear, XTest);      % Linear predictions

%plotting
figure;
hold on; 


plot(XTest, YTest, 'k-o', 'LineWidth', 3, 'MarkerSize', 6, 'DisplayName', 'Actual Data');

%all kernels plotting
plot(XTest, YPred_gaussian, 'b--s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Gaussian Optimized', 'MarkerIndices', 1:10:length(YPred_gaussian));
plot(XTest, YPred_rbf, 'r-.d', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'RBF Optimized', 'MarkerIndices', 1:10:length(YPred_rbf));
plot(XTest, YPred_poly, 'g:>', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Polynomial Order 2', 'MarkerIndices', 1:10:length(YPred_poly));
plot(XTest, YPred_linear, 'm--<', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Linear BC=10', 'MarkerIndices', 1:10:length(YPred_linear));


title('Comparison of SVR Kernel Predictions');
legend('Location', 'best');
xlabel('Time of Day');
ylabel('Traffic Flow');
grid on; % Add grid for better readability
hold off; % Release the plot

%save
saveas(gcf, 'CombinedKernelPredictions_Clean.png');
