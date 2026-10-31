import tensorflow as tf

# Same digits as before
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# CNN layers expect a "channel" number: 1 for grayscale (3 would be colour RGB)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),                       # 28x28 image, 1 channel
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu"),    # 8 filters of 3x3, then ReLU
    tf.keras.layers.MaxPooling2D((2, 2)),                    # shrink: keep max of each 2x2 block
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),   # 16 filters on the shrunk maps
    tf.keras.layers.MaxPooling2D((2, 2)),                    # shrink again
    tf.keras.layers.Flatten(),                               # unroll into one line
    tf.keras.layers.Dense(10, activation="softmax"),         # 10 digits, as percentages
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
model.evaluate(x_test, y_test)