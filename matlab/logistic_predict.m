% logistic prediction with wegiths and input data
function y=logistic_predict(weights, data)
    [N, M] = size(data);
    z = zeros(1,N);
    ws = weights(1:end-1);
    b = weights(end);
    
    for i = 1:N
        d = dot(data(i,1:end), ws); % dot product
        z(i) = d + b;
        y(i) = sigmoid(z(i)); % managing value with sigmoid function
    end
end