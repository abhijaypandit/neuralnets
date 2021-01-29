import keras

keras.backend.clear_session()

# Load dataset
dataset = keras.datasets.mnist
print("Loading dataset...")
(X_train, y_train), (X_test, y_test) = dataset.load_data()

# Normalise examples
X_train = X_train/255.0
X_test = X_test/255.0

# Define model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=X_train.shape[1:]),
    keras.layers.Dense(64, activation=keras.activations.relu),
    keras.layers.Dense(32, activation=keras.activations.relu),
    keras.layers.Dense(10, activation=keras.activations.softmax)
])
model.summary()

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