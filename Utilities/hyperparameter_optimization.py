# hyperparameter_optimization.py

from bayes_opt import BayesianOptimization
from NN_model import Wetland_NN

def my_optimisation(X_train, X_val, Y_train, Y_val):
    all_runs_details = []

    def train_evaluate_model(num_hidden_units, initial_learning_rate, dropout_rate, num_layers, batch_size):
        # Adjusted to match the provided param_dict, removing lstm_activation and dense_activation
        param_dict = {
            'numHiddenUnits': int(num_hidden_units),
            'initial_learning_rate': initial_learning_rate,
            'num_layers': int(num_layers),  # Make sure num_layers is treated as an integer
            'dropout_rate': None if dropout_rate <= 0 else dropout_rate,
            'batch_size': int(batch_size),  # Make sure batch_size is treated as an integer
            'num_epochs': 600,  # Updated to match provided param_dict
            'learning_rate_schedule': 'exponential',
            'optimizer': 'adam',
            'early_stopping_patience': 50,
        }
        net, history = Wetland_NN(X_train, X_val, Y_train, Y_val, **param_dict)
        best_val_loss = min(history.history['val_loss'])
        all_runs_details.append({
            'params': param_dict,
            'best_val_loss': best_val_loss
        })
        return -best_val_loss

    # Updated pbounds to reflect the actual parameters to be optimized
    pbounds = {
        'num_hidden_units': (40, 200),
        'initial_learning_rate': (1e-4, 1e-2),
        'num_layers': (1, 3),  # Adjusted to reasonable bounds
        'batch_size': (512, 512),  # Adjusted to reasonable bounds
        'dropout_rate': (0, 0),
    }

    optimizer = BayesianOptimization(
        f=train_evaluate_model,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=18,
        n_iter=10,
    )

    best_params = optimizer.max['params']
    return best_params, all_runs_details