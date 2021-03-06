from check_grad import check_grad
from utils import *
from logistic import *

def run_logistic_regression(hyperparameters):
    # TODO specify training data
    train_inputs, train_targets = load_train()

    valid_inputs, valid_targets = load_valid()

    # N is number of examples; M is the number of features per example.
    N, M = train_inputs.shape
    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.randn(M+1, 1)- 0.5

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters, weights, train_inputs, train_targets)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for t in range(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        print ("ITERATION:{%4d}  TRAIN NLOGL:{%4.2f}  TRAIN CE:{%.6f} TRAIN FRAC:{%2.2f}  VALID CE:{%.6f}  VALID FRAC:{%2.2f}" %
                   (t+1, f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100) )

        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100]
    return logging, weights

def run_check_grad(hyperparameters, weights, data, targets):
    """Performs gradient check on logistic function.
    """

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print ("diff =", diff)

if __name__ == '__main__':
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.01,
                    'weight_regularization': 1, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 10
                    }

    # average over multiple runs
    num_runs = 1    
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    weights = np.zeros((hyperparameters['num_iterations'], 100) )
    for i in range(num_runs):
        logging, weights = run_logistic_regression(hyperparameters)
    # TODO generate plots