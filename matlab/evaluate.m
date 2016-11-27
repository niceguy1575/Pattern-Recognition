function [ce, frac_correct] = evaluate(targets, y)
   
    ce = -dot(targets', log(y))- dot(1-targets', log(1-y));
    
    ts = size(targets);
    
    correct = 0;
    for i = 1:ts(1)
        if abs(targets(i) - y(i)) < 0.5
            correct = correct + 1;
        end
    end
    
    frac_correct = correct/ts(1);
end

