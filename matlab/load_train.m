function [train_inputs, train_targets] = load_train()
    % Loads training data.
        train_inputs = readNPY('data/train_inputs.npy'); 
        train_targets = readNPY('data/train_targets.npy'); 
end