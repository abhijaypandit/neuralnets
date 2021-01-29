import keras

keras.backend.clear_session()

def normalize(X):
    x = keras.utils.normalize(X, order=2) # L2 norm
    return x

def build_model(layers=1, neurons=32, learning_rate=0.01):
    # define model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    for _ in range(layers):
        model.add(keras.layers.Dense(neurons, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # compile model
    model.compile(
        loss=keras.losses.sparse_categorical_crossentropy,
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    return model

def train_model(model, X_train, y_train, h_params, callbacks):
    fit = model.fit(
        X_train, y_train,
        batch_size=h_params['batch_size'],
        epochs=h_params['epochs'],
        callbacks=callbacks,
        validation_split=0.2,
    )

    return fit

if __name__ == "__main__":
    # Load dataset
    dataset = keras.datasets.mnist
    print("Loading dataset...")
    (X_train, y_train), (X_test, y_test) = dataset.load_data()

    # Normalise examples
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # Define hyperparameters
    h_params = {}
    h_params['batch_size'] = 32
    h_params['epochs'] = 1000

    # Define callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

    callbacks = []
    callbacks.append(early_stop)

    # Build model
    model = build_model(layers=2, neurons=64)

    # Train model
    fit = train_model(model,X_train, y_train, h_params, callbacks)

    # Evaluate model
    model.evaluate(X_test, y_test)