import tensorflow as tf

# Define the TensorFlow neural network model
DiabetesTFModel = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
