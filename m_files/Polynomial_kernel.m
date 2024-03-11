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

% orders and box constraints list
polynomialOrders = 2; % Polynomial orders to 4iterate over
boxConstraints = 10; % we can expand this as needed

% looping over each po and bc
for i = 1:length(polynomialOrders)
    polynomialOrder = polynomialOrders(i);
    
    % Loop over each box constraint for the current polynomial order
    for j = 1:length(boxConstraints)
     
        boxConstraint = boxConstraints(j);

        % train svr based on current settings
        model_poly = fitrsvm(XTrain, YTrain, 'KernelFunction', 'polynomial', ...
                            'PolynomialOrder', polynomialOrder, 'BoxConstraint', boxConstraint);

        % Make predictions with the trained model
        YPred_poly = predict(model_poly, XTest);

        % metric
        mse_poly = immse(YPred_poly, YTest);
        r2_poly = 1 - sum((YTest - YPred_poly).^2) / sum((YTest - mean(YTest)).^2);


        fprintf('Polynomial Order: %d, Box Constraint: %d\n', polynomialOrder, boxConstraint);
        fprintf('MSE: %.4f, R2: %.4f\n', mse_poly, r2_poly);

        % plot
        figure;
        plot(XTest, YTest, 'bo', XTest, YPred_poly, 'r*');
        legend('Actual', 'Predicted');
        title(sprintf('SVR Predictions with Polynomial Order=%d, BoxConstraint=%d', polynomialOrder, boxConstraint));
        xlabel('Local Time (Numeric)');
        ylabel('Total Carriageway Flow');
        
    end
end

% optimize with the polynomial order between 2 and 5
optimizedVars = [
    optimizableVariable('PolynomialOrder',[2,3],'Type','integer'), ...
    optimizableVariable('BoxConstraint',[5,15],'Transform','log')
];


bestSet = struct('PolynomialOrder', [], 'BoxConstraint', [], 'MSE', Inf, 'R2', -Inf);

% Run Bayesian optimization multiple times
numIterations = 5; % number of iterations
for iter = 1:numIterations
    MdlOptimized = fitrsvm(XTrain, YTrain, ...
        'KernelFunction','polynomial', ...
        'OptimizeHyperparameters',optimizedVars, ...
        'HyperparameterOptimizationOptions',struct( ...
            'AcquisitionFunctionName','expected-improvement-plus', ...
            'ShowPlots',false, ... % Set to true to see plots during optimization
            'Verbose',0, ... % Set to 1 or 2 for more verbosity
            'Repartition',false, ...
            'UseParallel',false, ... % Set to true if you have Parallel Computing Toolbox
            'MaxObjectiveEvaluations',30)); % Increase if you need more evaluations

    % Extract the best hyperparameters
    bestParams = MdlOptimized.HyperparameterOptimizationResults.XAtMinObjective;
    YPred_optimized = predict(MdlOptimized, XTest);

    % =We calculate the MSE and R2 for the optimized model
    mse_optimized = immse(YPred_optimized, YTest);
    r2_optimized = 1 - sum((YTest - YPred_optimized).^2) / sum((YTest - mean(YTest)).^2);

    % see that if the performance is better, update the best hyperparameter set and performance
    if mse_optimized < bestSet.MSE
        bestSet.PolynomialOrder = bestParams.PolynomialOrder;
        bestSet.BoxConstraint = bestParams.BoxConstraint;
        bestSet.MSE = mse_optimized;
        bestSet.R2 = r2_optimized;
    end
end

%display code
disp('Best Hyperparameters from Optimization:');
disp(bestSet);

figure;
scatter(XTest, YTest, 'filled');
hold on;
scatter(XTest, YPred_optimized, 'd');
hold off;
legend('Actual', 'Predicted');
title(sprintf('SVR Predictions with Optimized Polynomial Kernel (Order: %d, Constraint: %.3f)', ...
    bestSet.PolynomialOrder, bestSet.BoxConstraint));
xlabel('Local Time (Numeric)');
ylabel('Total Carriageway Flow');
grid on;

