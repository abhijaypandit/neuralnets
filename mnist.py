import keras

keras.backend.clear_session()

# Load dataset
dataset = keras.datasets.mnist
print("Loading dataset...")
(X_train, y_train), (X_test, y_test) = dataset.load_data()

# Normalise examples
X_train = keras.utils.normalize(X_train, order=2)
X_test = keras.utils.normalize(X_test, order=2)

# Define model
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation=keras.activations.relu),
    keras.layers.Dense(32, activation=keras.activations.relu),
    keras.layers.Dense(10, activation=keras.activations.softmax)
])
# model.summary() -- no input_shape given to model

# Compile model
model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    optimizer=keras.optimizers.SGD(),
    metrics=['accuracy']
)

# Train model
fit = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2
)

# Evaluate model
model.evaluate(X_test, y_test)