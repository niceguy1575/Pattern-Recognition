function run_check_grad(hyperparameters, weights, data, targets)
    % Performs gradient check on logistic function.

    diff = check_grad(weights, 0.001, data, targets, hyperparameters);
    fprintf('diff = %f \n', diff);
end