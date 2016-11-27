% get hyper parameter
regularized = {0.001, 0.01, 0.1, 1.0};
lambda = {0.001, 0.01, 0.1, 1.0};
hyperparameters = struct('learning_rate', lambda, 'num_iterations', 100, 'weight_regularization', regularized );
train_type = struct('all', 1, 'normal', 0, 'small', 0);
logging = zeros(getfield(hyperparameters,'num_iterations') * length(regularized) * length(regularized), 5);

% we don't use weights : hyperparameter
logging_avg = zeros(length(regularized), length(lambda));
number = 1:100:1700;

for i = 1: length(regularized)
    for j = 1:length(lambda)
        hyperparameters = struct('learning_rate', lambda(i), 'num_iterations', 100, 'weight_regularization', regularized(j) );
        index = 4*(i-1) + j;
        [logging(number(index):number(index+1)-1,:)] = run_logistic_regression_regularized(train_type, hyperparameters);
        logging_avg(i,j) = mean(logging(number(index):number(index+1)-1,4));
    end
end

[minval,ind] = min(logging_avg(:));
[I,J] = ind2sub([size(logging_avg,1) size(logging_avg,2)],ind);
index = 4*(I-1) + J;
logging = logging(number(index):number(index+1)-1,:);

hyper_regularized = regularized(I);
hyper_lambda = lambda(J);

% figure : validation ce diff(1) and frac correct diff(1)
figure
subplot(2,1,1);
plot(diff(logging(:,4)));

subplot(2,1,2);
plot(diff(logging(:,5)));

% get the hyper_iteration
t = diff(logging(:,4)) > -0.15 ;
hyper_iteration = find(t,1);

% logistic with hyper_iteration
hyperparameters = struct('learning_rate', hyper_lambda, 'num_iterations', hyper_iteration, 'weight_regularization', hyper_regularized);

logging_train_normal = zeros(getfield(hyperparameters,'num_iterations'), 5);
weights_train_normal = zeros(getfield(hyperparameters,'num_iterations'), 785);
logging_train_small = zeros(getfield(hyperparameters,'num_iterations'), 5);
weights_train_small = zeros(getfield(hyperparameters,'num_iterations'), 785);

train_type = struct('all', 0, 'normal', 1, 'small', 0);
[logging_train_normal, weights_train_normal] = run_logistic_regression_regularized(train_type, hyperparameters);

train_type = struct('all', 0, 'normal', 0, 'small', 1);
[logging_train_small, weights_train_small] = run_logistic_regression_regularized(train_type, hyperparameters);

% error graph : train_normal
figure
subplot(2,1,1);
plot(logging_train_normal(:,4));

subplot(2,1,2);
plot(logging_train_normal(:,5));

% error graph : train_small
figure
subplot(2,1,1);
plot(logging_train_small(:,4));

subplot(2,1,2);
plot(logging_train_small(:,5));

% test classification by regularization weight
logging_final = zeros(getfield(hyperparameters,'num_iterations'), 5);
weights_final = zeros(getfield(hyperparameters,'num_iterations'), 785);

% other hyper parameters are fixed
test_error_rate = zeros(1, length(regularized));
hyperparameters = struct('learning_rate', hyper_lambda, 'num_iterations', hyper_iteration, 'weight_regularization', regularized);

[test_inputs, test_targets] = load_test();
szt = size(test_targets);
train_type = struct('all', 1, 'normal', 0, 'small', 0);

for i = 1:length(regularized)
    [logging_final, weights_final] = run_logistic_regression_regularized(train_type, hyperparameters(i) );

    test_res = logistic_predict(weights_final, test_inputs);

    test_res(test_res>0.5) = 1;
    test_res(test_res<=0.5) = 0;

    % test error rate
    test_error_rate(i) = sum(test_res' == test_targets) / szt(1);
end

% Test Error Rate for hyperparameter : 10 trial

num_runs = 10;
test_error_rate_final = zeros(1, num_runs);
hyperparameters = struct('learning_rate', hyper_lambda, 'num_iterations', hyper_iteration, 'weight_regularization', hyper_regularized);

for i = 1:num_runs
    [logging_final, weights_final] = run_logistic_regression_regularized(train_type, hyperparameters);

    test_res = logistic_predict(weights_final, test_inputs);

    test_res(test_res>0.5) = 1;
    test_res(test_res<=0.5) = 0;

    % test error rate
    test_error_rate_final(i) = sum(test_res' == test_targets) / szt(1);
end