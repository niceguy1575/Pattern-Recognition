function [test_inputs, test_targets] = load_test()
    % Loads test data.
        test_inputs = readNPY('data/test_inputs.npy'); 
        test_targets = readNPY('data/test_targets.npy'); 
end
