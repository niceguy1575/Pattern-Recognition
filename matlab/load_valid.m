function [test_inputs, test_targets] = load_valid()
    % Loads validation data.
        test_inputs = readNPY('data/valid_inputs.npy'); 
        test_targets = readNPY('data/valid_targets.npy'); 
end