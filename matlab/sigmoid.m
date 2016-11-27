function s = sigmoid(x)
    %Computes the element wise logistic sigmoid of x.

    %Inputs:
    %    x: Either a row vector or a column vector.
    
    s = 1 / (1.0 + exp(-x) );
    
end