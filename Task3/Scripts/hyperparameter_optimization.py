# hyperparameter_optimization.py

from bayes_opt import BayesianOptimization
from NN_model import Wetland_NN

def my_optimisation(X_train, X_val, Y_train, Y_val):
    all_runs_details = []

    def train_evaluate_model(num_hidden_units, initial_learning_rate, dropout_rate, num_layers, batch_size, lstm_activation, dense_activation):
        param_dict = {
            'numHiddenUnits': int(num_hidden_units),
            'initial_learning_rate': initial_learning_rate,
            'num_layers': 2,
            'dropout_rate': None if dropout_rate <= 0 else dropout_rate,
            'batch_size': 32,
            'num_epochs': 50,
            'learning_rate_schedule': 'exponential',
            'optimizer': 'adam',
            'early_stopping_patience': 50,
            'lstm_activation': 'tanh' if lstm_activation < 0.5 else 'relu',
            'dense_activation': 'linear' if dense_activation < 0.5 else 'relu'
        }
        net, history = Wetland_NN(X_train, X_val, Y_train, Y_val, **param_dict)
        best_val_loss = min(history.history['val_loss'])
        all_runs_details.append({
            'params': param_dict,
            'best_val_loss': best_val_loss
        })
        return -best_val_loss

    pbounds = {
        'num_hidden_units': (100, 400),
        'initial_learning_rate': (1e-4, 1e-2),
        'num_layers': (1,2),
        'batch_size': (32,128),
        'dropout_rate': (0, 0.5),
        'lstm_activation': (0, 0),
        'dense_activation': (0, 0)
    }

    optimizer = BayesianOptimization(
        f=train_evaluate_model,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=10,
    )

    best_params = optimizer.max['params']
    return best_params, all_runs_details
