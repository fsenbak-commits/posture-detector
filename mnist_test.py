import tensorflow as tf  # TensorFlow; Keras (tf.keras) is its easy network-building part

# Load the handwritten digits: 60,000 training images + 10,000 test images
# x = the images (28x28 pixels, values 0-255), y = the correct answers (0-9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalise: scale pixels from 0-255 down to 0-1 (networks learn better with small numbers)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the network: layers stacked in order, each feeding the next
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),                    # each input is a 28x28 image
    tf.keras.layers.Flatten(),                         # unroll the 28x28 grid into 784 numbers (input layer)
    tf.keras.layers.Dense(16, activation="relu"),   # hidden layer 1: 16 neurons, each connected to all 784 inputs
    tf.keras.layers.Dense(16, activation="relu"),   # hidden layer 2: 16 neurons
    tf.keras.layers.Dense(10, activation="softmax"),   # output layer: 10 neurons (digits 0-9), as probabilities summing to 1
])

# Choose how to learn:
#   optimizer = how to walk downhill (Adam = improved gradient descent)
#   loss      = the cost function (how wrong the network is)
#   metrics   = also report accuracy (% correct), easier for humans to read
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Print the layers and how many weights + biases each has (total should be 13,002)
model.summary()

# Train: forward pass + backpropagation + gradient descent, all automatic
#   epochs=10           -> go through all training images 10 times
#   batch_size=32       -> one gradient step per 32 images (mini-batches)
#   validation_split    -> keep 10% aside to check progress on unseen images
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Final exam: test on 10,000 images the network has never seen
model.evaluate(x_test, y_test)