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

boxConstraints = [250];

% we loop over each box constraint for the linear kernel
for j = 1:length(boxConstraints)
    boxConstraint = boxConstraints(j);

    % training SVR model with the linear kernel and current settings
    model_linear = fitrsvm(XTrain, YTrain, 'KernelFunction', 'linear', ...
                           'BoxConstraint', boxConstraint);
    YPred_linear = predict(model_linear, XTest);

    % mse r2
    mse_linear = immse(YPred_linear, YTest);
    r2_linear = 1 - sum((YTest - YPred_linear).^2) / sum((YTest - mean(YTest)).^2);

    fprintf('Linear Kernel with BoxConstraint=%d\n', boxConstraint);
    fprintf('MSE: %.4f, R2: %.4f\n', mse_linear, r2_linear);

    % Oplot results
    figure;
    plot(XTest, YTest, 'bo', XTest, YPred_linear, 'r*');
    legend('Actual', 'Predicted');
    title(sprintf('SVR Predictions with Linear Kernel, BoxConstraint=%d', boxConstraint));
    xlabel('Local Time (Numeric)');
    ylabel('Total Carriageway Flow');

    saveas(gcf, sprintf('SVR_Linear_BoxConstraint%d.png', boxConstraint));
    
end


%%%%%%
% hyperparameter optimization
optimizedVars = optimizableVariable('BoxConstraint',[1e-3,1e3],'Transform','log');
bestSet = struct('BoxConstraint', [], 'MSE', Inf, 'R2', -Inf);

% Run Bayesian optimization multiple times
numIterations = 5; %iter count
for iter = 1:numIterations
    MdlOptimized = fitrsvm(XTrain, YTrain, ...
        'KernelFunction','linear', ...
        'OptimizeHyperparameters',optimizedVars, ...
        'HyperparameterOptimizationOptions',struct( ...
            'AcquisitionFunctionName','expected-improvement-plus', ...
            'ShowPlots',false, ... 
            'Verbose',0, ... 
            'Repartition',false, ...
            'UseParallel',false, ... 
            'MaxObjectiveEvaluations',30)); 

    % best params
    bestParams = MdlOptimized.HyperparameterOptimizationResults.XAtMinObjective;
    YPred_optimized = predict(MdlOptimized, XTest);

    % mse r2
    mse_optimized = immse(YPred_optimized, YTest);
    r2_optimized = 1 - sum((YTest - YPred_optimized).^2) / sum((YTest - mean(YTest)).^2);

    % If the performance is better then update 
    if mse_optimized < bestSet.MSE
        bestSet.BoxConstraint = bestParams.BoxConstraint;
        bestSet.MSE = mse_optimized;
        bestSet.R2 = r2_optimized;
    end
end

% Disp the best hyperparameters and performance
disp('Best Hyperparameters from Optimization:');
disp(bestSet);

% Plotting code
figure;
scatter(XTest, YTest, 'filled');
hold on;
scatter(XTest, YPred_optimized, 'd');
hold off;
legend('Actual', 'Predicted');
title(sprintf('SVR Predictions with Optimized Linear Kernel, BoxConstraint=%.3f', ...
    bestSet.BoxConstraint));
xlabel('Local Time (Numeric)');
ylabel('Total Carriageway Flow');
grid on;

