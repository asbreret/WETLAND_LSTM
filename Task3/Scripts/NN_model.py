import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay

def Wetland_NN(
    X_train, 
    X_val, 
    Y_train, 
    Y_val, 
    numHiddenUnits=50, 
    initial_learning_rate=0.01,  # make sure this line is present
    num_epochs=200,
    num_layers=1, 
    dropout_rate=None, 
    batch_size=256, 
    learning_rate_schedule='exponential', 
    optimizer='adam', 
    early_stopping_patience=100, 
    lstm_activation='tanh', 
    dense_activation='linear'):
    """
    This function constructs an LSTM neural network for the Fish tank data.
    """
    # Determine maximum sequence length
    seq_length_x = X_train.shape[1]
    seq_length_y = Y_train.shape[1]

    # Define the LSTM network architecture
    numFeatures = X_train.shape[2]
    numResponses = Y_train.shape[2]

    # Define the model
    model = Sequential()
    model.add(Input(shape=(seq_length_x, numFeatures)))
    for _ in range(num_layers):
        model.add(Bidirectional(LSTM(numHiddenUnits, return_sequences=True, activation=lstm_activation)))
        if dropout_rate is not None:
            model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(seq_length_y * numResponses, activation=dense_activation))
    model.add(Reshape((seq_length_y, numResponses)))

    # Define a learning rate schedule
    if learning_rate_schedule == 'exponential':
        lr_schedule = ExponentialDecay(initial_learning_rate=initial_learning_rate, decay_steps=1000, decay_rate=0.5)
    elif learning_rate_schedule == 'polynomial':
        lr_schedule = PolynomialDecay(initial_learning_rate=initial_learning_rate, decay_steps=1000, end_learning_rate=0.01)
    else:
        lr_schedule = initial_learning_rate

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)

    # Create an optimizer with the learning rate schedule
    if optimizer == 'adam':
        optimizer_obj = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif optimizer == 'rmsprop':
        optimizer_obj = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    elif optimizer == 'adagrad':
        optimizer_obj = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
    else:
        raise ValueError('Unknown optimizer')

    # Compile the model
    model.compile(loss='mse', optimizer=optimizer_obj)
    
    # Fit the model
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=num_epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])

    return model, history