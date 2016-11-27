% Methods for doing logistic regression.
function [f, df, y] = logistic(weights, data, targets, hyperparameters)

    [N, M] = size(data);
    z = zeros(1,N);
    ws = weights(1:end-1);
    b = weights(end);
    for i = 1:N
        d = dot(data(i,1:end), ws);
        z(i) = d + b;
        zg(i) = sigmoid(z(i));
    end
    
    df = zeros(1,M+1);
    
    for j = 1:M
        df(j) = dot(data(:, j)', (zg' - targets));
    end
    
    aa = (1 - targets);
    b = dot(zg, -exp(-z)) + dot(ones(N,1), aa);

    df(M+1) = b;
    y = logistic_predict(weights, data);
    fs = evaluate(targets, y);
    f = fs(1);
end