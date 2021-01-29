import time
import keras
import matplotlib.pyplot as plt

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
        verbose=False
    )

    return fit

def plot(train, metric, name="Figure"):
    fig = plt.figure()
    plt.title(name)
    for x in metric:
        plt.plot(train.history[x], label=x)
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    return fig

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
    model = build_model(layers=2, neurons=64, learning_rate=0.03)

    # Train model
    print("Training model...")
    start = time.time()
    train = train_model(model, X_train, y_train, h_params, callbacks)
    end = time.time()

    model.summary()

    print("Training time = {:.2f}s".format(end-start))
    print("Loss = {:.2f}".format(train.history['loss'][-1]))
    print("Accuracy = {:.2f}".format(train.history['accuracy'][-1]*100))

    fig = plot(train, ['loss', 'val_loss', 'accuracy', 'val_accuracy'], name="Performance")

    # Evaluate model
    print("\nEvaluating model...")
    eval_loss, eval_accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Loss = {:.2f}\nAccuracy = {:.2f}".format(eval_loss, eval_accuracy*100))

    fig.show()
    