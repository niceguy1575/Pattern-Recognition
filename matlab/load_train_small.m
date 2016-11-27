function [train_inputs_small, train_targets_small] = load_train_small()
    % Loads small training data.
        train_inputs_small =  readNPY('data/train_small_inputs.npy');
        train_targets_small =  readNPY('data/train_small_targets.npy');
end