function [logging, weights] = run_logistic_regression_regularized(train_type, hyperparameters)
    
    % load other data type train_type
    if getfield(train_type,'normal') == 1
        [train_inputs, train_targets] = load_train();
    elseif getfield(train_type,'small') == 1
        [train_inputs, train_targets] = load_train_small();
        % file train_small_target dimension error : multiplied by 2
        train_targets = [0,0,0,0,0,1,1,1,1,1]';
    elseif getfield(train_type,'all') == 1
        [train_inputs, train_targets] = load_train();
        [train_small_inputs, train_small_targets] = load_train_small();
        
        train_small_targets = [0,0,0,0,0,1,1,1,1,1]';
        train_inputs = vertcat(train_inputs, train_small_inputs);
        train_targets = vertcat(train_targets, train_small_targets);
    else
        ;
    end
    
    [valid_inputs, valid_targets] = load_valid();

    % N is number of examples; M is the number of features per example.
    [N, M] = size(train_inputs);
    % Logistic regression weights : random weights
    weights = rand(M+1, 1) - 0.5;

    % Verify that your logistic function produces the right gradient.
    % diff should be very close to 0.
    run_check_grad(hyperparameters, weights, train_inputs, train_targets);

    % Begin learning with gradient descent
    logging = zeros( getfield(hyperparameters,'num_iterations'), 5) ;
    for t = 1:getfield(hyperparameters,'num_iterations')

        % penalized logistic regression
        [f, df, predictions] = logistic_pen(weights, train_inputs, train_targets, hyperparameters);
        
        % Evaluate the prediction.
        [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions);

        if isnan(f) || isinf(f)
            disp( 'ValueError("nan/inf error")' );
            logging(t,:) = [nan, nan, nan, nan, nan];
            continue;
        end
        
        % update parameters
        weights = weights - (getfield(hyperparameters,'learning_rate') * df / N)';
        
        % Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs);

        % Evaluate the prediction.
        [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
        
        logging(t,:) = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100];
    end
end