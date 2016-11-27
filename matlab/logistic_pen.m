function [f, df, y] = logistic_pen(weights, data, targets, hyperparameters)

    [N, M] =size(data);
    y = logistic_predict(weights,data);
    alpha = getfield(hyperparameters,'weight_regularization');
    
    z = zeros(1,N);
    ws = weights(1:end-1);
    b = weights(end);
    for i = 1:N
        d = dot(data(i,1:end), ws);
        z(i) = d + b;
        zg(i) = sigmoid(z(i));
    end
    % f is the loss function plus lambda/2 * sigma (wi^2) : lw penalized
    fs = evaluate(targets,y);
    f =  fs(1) + 0.5 * alpha * (dot(weights',weights));
    % df is the final simplying after gradient and differentiation plus lambda wi
    % df = dot(cmp_data',(1-targets)-(1-y)) + alpha * weights;
    
    df = zeros(1,M+1);
    
    for j = 1:M
        df(j) = dot(data(:, j)', (zg' - targets)) + alpha * weights(j);
    end
    
    aa = (1 - targets);
    b = dot(zg, -exp(-z)) + dot(ones(N,1), aa);

    df(M+1) = b;
end
