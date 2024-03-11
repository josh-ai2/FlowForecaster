% load
trainData = readtable('trainData.csv');
testData = readtable('testData.csv');

% localtime
trainData.Local_Time_Numeric = hours(duration(trainData.('LocalTime')));
testData.Local_Time_Numeric = hours(duration(testData.('LocalTime')));

XTrain = trainData.Local_Time_Numeric;
YTrain = trainData.TotalCarriagewayFlow;
XTest = testData.Local_Time_Numeric;
YTest = testData.TotalCarriagewayFlow;

% kernel scales
kernelScales = {'auto', 5, 10}; % Including 'auto' as a string and numeric values
boxConstraints = [1, 10, 100]; % Box constraints to iterate over

for i = 1:length(kernelScales)
    kernelScale = kernelScales{i};
    boxConstraint = boxConstraints(i);
    
    % Train the SVR model with the current settings
    model_rbf = fitrsvm(XTrain, YTrain, 'KernelFunction', 'rbf', ...
                        'KernelScale', kernelScale, 'BoxConstraint', boxConstraint);
    
    % prediction part
    YPred_rbf = predict(model_rbf, XTest);
    
    % display metrtics
    mse_rbf = immse(YPred_rbf, YTest);
    r2_rbf = 1 - sum((YTest - YPred_rbf).^2) / sum((YTest - mean(YTest)).^2);
    results_rbf = table(mse_rbf, r2_rbf, 'VariableNames', {'MSE', 'R2'});
    disp(results_rbf);
    
    % plotting
    figure;
    plot(XTest, YTest, 'bo', XTest, YPred_rbf, 'r*');
    legend('Actual', 'Predicted');
    title(sprintf('SVR Predictions with KernelScale=%s, BoxConstraint=%d', string(kernelScale), boxConstraint));
    xlabel('Local Time (Numeric)');
    ylabel('Total Carriageway Flow');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hyperparameter optimization
optimizedVars = [
    optimizableVariable('KernelScale',[1e-3,1e3],'Transform','log'), ...
    optimizableVariable('BoxConstraint',[1e-3,1e3],'Transform','log')
];

% storing the values
bestSet = struct('KernelScale', [], 'BoxConstraint', [], 'MSE', Inf, 'R2', -Inf);

% Run Bayesian optimization multiple times
numIterations = 5; % iteration count
for iter = 1:numIterations
    MdlOptimized = fitrsvm(XTrain, YTrain, ...
        'KernelFunction','rbf', ...
        'OptimizeHyperparameters',optimizedVars, ...
        'HyperparameterOptimizationOptions',struct( ...
            'AcquisitionFunctionName','expected-improvement-plus', ...
            'ShowPlots',false, ... 
            'Verbose',0, ... 
            'Repartition',false, ...
            'UseParallel',false, ... 
            'MaxObjectiveEvaluations',30)); 

    % best params extract
    bestParams = MdlOptimized.HyperparameterOptimizationResults.XAtMinObjective;

    % optimized settings, use predict
    YPred_optimized = predict(MdlOptimized, XTest);

    % MSE and R2 for the optimized model
    mse_optimized = immse(YPred_optimized, YTest);
    r2_optimized = 1 - sum((YTest - YPred_optimized).^2) / sum((YTest - mean(YTest)).^2);

    %if case:if better then update the best hyperparameter set and performance
    if mse_optimized < bestSet.MSE
        bestSet.KernelScale = bestParams.KernelScale;
        bestSet.BoxConstraint = bestParams.BoxConstraint;
        bestSet.MSE = mse_optimized;
        bestSet.R2 = r2_optimized;
    end
end

% display and plotting code
disp('Best Hyperparameters from Optimization:');
disp(bestSet);

figure;
scatter(XTest, YTest, 'filled');
hold on;
scatter(XTest, YPred_optimized, 'd');
hold off;
legend('Actual', 'Predicted');
title(sprintf('SVR Predictions with Optimized RBF Kernel (Scale: %.3f, Constraint: %.3f)', ...
    bestSet.KernelScale, bestSet.BoxConstraint));
xlabel('Local Time (Numeric)');
ylabel('Total Carriageway Flow');
grid on;



