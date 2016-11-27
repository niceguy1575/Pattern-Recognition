function d = check_grad(X, epsilon, data, targets, hyperparameters)
    
    % Centered Difference Qutoient
    szx = size(X);
    if length(szx) ~= 2  || szx(2) ~= 1
        error( 'ValueError(X must be a vector)');
    end
    
    [y, dy, z] = logistic(X, data, targets, hyperparameters);
    
    dh = zeros(length(X), 1);

    for j = 1:length(X)
        dx = zeros(length(X), 1);
        dx(j) = epsilon;
        [y2, dy2, z2] = logistic(X+dx, data, targets, hyperparameters);
        dx = -dx;
        [y1 dy1 z1] = logistic(X+dx, data, targets, hyperparameters);
        dh(j) = (y2 - y1)/(2*epsilon);
    end
    d = norm(dh-dy) / norm(dh+dy);  % return norm of diff divided by norm of sum
end
