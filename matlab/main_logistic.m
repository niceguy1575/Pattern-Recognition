% get the hyper parameters - learning rate and iteration
lambda = {0.001, 0.01, 0.1, 1.0};
hyperparameters = struct('learning_rate', lambda, 'num_iterations', 100);
train_type = struct('all', 1, 'normal', 0, 'small', 0);

logging = zeros(getfield(hyperparameters,'num_iterations') * length(lambda), 5);

% In this case, we don't use weights
logging_avg = zeros(1, length(lambda));

for i = 1:length(lambda)
    start_num = (getfield(hyperparameters,'num_iterations')*(i-1)+1);
    end_num = getfield(hyperparameters,'num_iterations')*i;
    [logging(start_num:end_num,:)] = run_logistic_regression(train_type, hyperparameters(i));
    logging_avg(i) = mean(logging(start_num:end_num,4));
end

[minimum, index] = min(logging_avg);
hyper_lambda = lambda(index);

start_num = (getfield(hyperparameters,'num_iterations')*(index-1)+1);
end_num = getfield(hyperparameters,'num_iterations')*index;

logging = logging(start_num:end_num,:);

% figure : validation ce diff(1) and frac correct diff(1)
figure
subplot(2,1,1);
plot(diff(logging(:,4)));

subplot(2,1,2);
plot(diff(logging(:,5)));

% get the hyper_iteration : -0.15 is the margin
t = diff(logging(:,4)) > -0.15 ;
hyper_iteration = find(t,1);

% logistic with hyper parameters
hyperparameters = struct('learning_rate', hyper_lambda, 'num_iterations', hyper_iteration);

logging_train_normal = zeros(getfield(hyperparameters,'num_iterations'), 5);
weights_train_normal = zeros(getfield(hyperparameters,'num_iterations'), 785);
logging_train_small = zeros(getfield(hyperparameters,'num_iterations'), 5);
weights_train_small = zeros(getfield(hyperparameters,'num_iterations'), 785);

% logistic regression with train_normal
train_type = struct('all', 0, 'normal', 1, 'small', 0);
[logging_train_normal, weights_train_normal] = run_logistic_regression(train_type, hyperparameters);

% logistic regression with train_small
train_type = struct('all', 0, 'normal', 0, 'small', 1);
[logging_train_small, weights_train_small] = run_logistic_regression(train_type, hyperparameters);

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

% test classification
logging_final = zeros(getfield(hyperparameters,'num_iterations'), 5);
weights_final = zeros(getfield(hyperparameters,'num_iterations'), 785);

num_runs = 10; %number of runs
test_error_rate = zeros(1, num_runs);

[test_inputs, test_targets] = load_test();
szt = size(test_targets);
train_type = struct('all', 1, 'normal', 0, 'small', 0); %use all train data

for i = 1:num_runs   
    [logging_final, weights_final] = run_logistic_regression(train_type, hyperparameters);

    test_res = logistic_predict(weights_final, test_inputs);
    
    % threshold : 0.5
    test_res(test_res>0.5) = 1;
    test_res(test_res<=0.5) = 0;

    % test error rate
    test_error_rate(i) = sum(test_res' == test_targets) / szt(1);
end