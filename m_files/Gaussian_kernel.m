% loading
trainData = readtable('trainData.csv');
testData = readtable('testData.csv');

% localtime
trainData.Local_Time_Numeric = hours(duration(trainData.('LocalTime')));
testData.Local_Time_Numeric = hours(duration(testData.('LocalTime')));

XTrain = trainData.Local_Time_Numeric;
YTrain = trainData.TotalCarriagewayFlow;
XTest = testData.Local_Time_Numeric;
YTest = testData.TotalCarriagewayFlow;


XTrain = XTrain(:);
XTest = XTest(:);

% Dcases
settings = [
    struct('KernelScale', 1, 'BoxConstraint', 1),     % Case 1
    struct('KernelScale', 5, 'BoxConstraint', 10),    % Case 2
    struct('KernelScale', 10, 'BoxConstraint', 100)   % Case 3
];

% storing of results
results = table('Size', [0, 4], 'VariableTypes', {'string', 'double', 'double', 'string'}, 'VariableNames', {'Kernel', 'MSE', 'R2', 'Details'});

%we look through each setting
for i = 1:length(settings)
    %trainig
    model = fitrsvm(XTrain, YTrain, 'KernelFunction', 'gaussian', ...
                    'KernelScale', settings(i).KernelScale, ...
                    'BoxConstraint', settings(i).BoxConstraint, ...
                    'Standardize', true);
    
    % preds
    YPred = predict(model, XTest);
    
    % metric
    mse = immse(YPred, YTest);
    r2 = 1 - sum((YTest - YPred).^2) / sum((YTest - mean(YTest)).^2);
    
    %Store results
    results = [results; {sprintf('Gaussian (Scale: %d, Constraint: %d)', ...
                settings(i).KernelScale, settings(i).BoxConstraint), mse, r2, ''}];
end


disp(results);

% Loop through each setting
for i = 1:length(settings)
    model = fitrsvm(XTrain, YTrain, 'KernelFunction', 'gaussian', ...
                    'KernelScale', settings(i).KernelScale, ...
                    'BoxConstraint', settings(i).BoxConstraint, ...
                    'Standardize', true);
    
    % Make predictions
    YPred = predict(model, XTest);
    
    % Calculate metrics
    mse = immse(YPred, YTest);
    r2 = 1 - sum((YTest - YPred).^2) / sum((YTest - mean(YTest)).^2);
    
    % Store results
    results = [results; {sprintf('Gaussian (Scale: %d, Constraint: %d)', ...
                settings(i).KernelScale, settings(i).BoxConstraint), mse, r2, ''}];
    
    % Plot the results for each setting
    figure;
    scatter(XTest, YTest, 'filled');
    hold on;
    scatter(XTest, YPred, 'd');
    hold off;
    legend('Actual', 'Predicted');
    title(sprintf('SVR Predictions with Gaussian Kernel (Scale: %d, Constraint: %d)', ...
        settings(i).KernelScale, settings(i).BoxConstraint));
    xlabel('Hour of Day');
    ylabel('Total Carriageway Flow');
    grid on;
end

%%%%%%%%%%%%%%
% hyperparam optimization
optimizedVars = [
    optimizableVariable('KernelScale',[1e-3,1e3],'Transform','log'), ...
    optimizableVariable('BoxConstraint',[1e-3,1e3],'Transform','log')
];

% bayes
MdlOptimized = fitrsvm(XTrain, YTrain, ...
    'KernelFunction','gaussian', ...
    'OptimizeHyperparameters',optimizedVars, ...
    'HyperparameterOptimizationOptions',struct( ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'ShowPlots',false, ... 
        'Verbose',0, ...
        'Repartition',false, ...
        'UseParallel',false, ... 
        'MaxObjectiveEvaluations',30)); 

% best params extraction
bestParams = MdlOptimized.HyperparameterOptimizationResults.XAtMinObjective;
YPred_optimized = predict(MdlOptimized, XTest);

% e MSE and R2 for the optimized model
mse_optimized = immse(YPred_optimized, YTest);
r2_optimized = 1 - sum((YTest - YPred_optimized).^2) / sum((YTest - mean(YTest)).^2);

% store
newRow = {sprintf('Gaussian Optimized'), mse_optimized, r2_optimized, ...
          sprintf('KernelScale: %.3f, BoxConstraint: %.3f', ...
          bestParams.KernelScale, bestParams.BoxConstraint)};

% Append the new row to the results table
results = [results; newRow];
disp(results);

% Plot results
figure; 
scatter(XTest, YTest, 'filled');
hold on; 
scatter(XTest, YPred_optimized, 'd');
hold off; 
legend('Actual', 'Predicted'); 
title('SVR Predictions with Optimized Gaussian Kernel'); 
xlabel('Hour of Day');
ylabel('Total Carriageway Flow'); 
grid on; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% grid search range
kernelScaleGrid = [0.1, 0.5, 1, 2, 5];
boxConstraintGrid = [0.1, 1, 10, 50, 100];

% results table 
resultsGrid = table('Size', [0, 4], 'VariableTypes', {'string', 'double', 'double', 'string'}, 'VariableNames', {'Kernel', 'MSE', 'R2', 'Details'});

% Grid Search
disp('Starting Grid Search...');
tic; % Start timing
for ks = kernelScaleGrid
    for bc = boxConstraintGrid
        model = fitrsvm(XTrain, YTrain, 'KernelFunction', 'gaussian', ...
            'KernelScale', ks, 'BoxConstraint', bc, 'Standardize', true);
        YPred = predict(model, XTest);
        mse = immse(YPred, YTest);
        r2 = 1 - sum((YTest - YPred).^2) / sum((YTest - mean(YTest)).^2);
        newRow = {sprintf('Gaussian (Scale: %.2f, Constraint: %.2f)', ks, bc), mse, r2, ''};
        resultsGrid = [resultsGrid; newRow];
    end
end
gridSearchTime = toc; % Stop timing

% Display Grid Search results and time
disp('Grid Search Results:');
disp(resultsGrid);
disp(['Grid Search Training Time: ', num2str(gridSearchTime), ' seconds']);

% Random Search
disp('Starting Random Search...');
randomSearchResults = resultsGrid(1,:); 
randomSearchResults(1,:) = []; 
numRandomTrials = 10; 
tic; % Start timing
for i = 1:numRandomTrials
    ks = randsample(kernelScaleGrid, 1);
    bc = randsample(boxConstraintGrid, 1);
    model = fitrsvm(XTrain, YTrain, 'KernelFunction', 'gaussian', ...
            'KernelScale', ks, 'BoxConstraint', bc, 'Standardize', true);
    YPred = predict(model, XTest);
    mse = immse(YPred, YTest);
    r2 = 1 - sum((YTest - YPred).^2) / sum((YTest - mean(YTest)).^2);
    newRow = {sprintf('Gaussian (Random Scale: %.2f, Constraint: %.2f)', ks, bc), mse, r2, ''};
    randomSearchResults = [randomSearchResults; newRow];
end
randomSearchTime = toc; % Stop timing

% Display Random Search results and time
disp('Random Search Results:');
disp(randomSearchResults);
disp(['Random Search Training Time: ', num2str(randomSearchTime), ' seconds']);






