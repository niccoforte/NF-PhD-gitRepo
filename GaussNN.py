import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def gaussian_mixture_model(inputs, num_components=3):
    # Assuming inputs have 2 dimensions (mean and log_variance)
    mean, log_variance = tf.split(inputs, 2, axis=-1)

    # Calculate variance from log_variance
    variance = tf.exp(log_variance)

    # Create a Lambda layer to sample from the Gaussian distribution
    def sampling(args):
        mean, variance = args
        batch_size = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        epsilon = K.random_normal(shape=(batch_size, dim))
        return mean + K.sqrt(variance) * epsilon

    samples = Lambda(sampling, output_shape=(num_components,), name='sample')([mean, variance])

    # Create a mixture density network output layer
    outputs = Dense(num_components * 3, activation='linear')(inputs)

    # Reshape the output to separate means, variances, and weights for each component
    outputs = Reshape((num_components, 3))(outputs)

    model = Model(inputs, outputs, name='gaussian_mixture_model')
    return model

# Create a simple example model
input_dim = 10  # Adjust based on your input dimensions
num_components = 3
inputs = Input(shape=(input_dim,), name='input')
outputs = gaussian_mixture_model(inputs, num_components=num_components)

# Compile the model
optimizer = tf.keras.optimizers.Adam(lr=0.001)
outputs.compile(optimizer=optimizer, loss='mean_squared_error')  # Adjust the loss function as needed

# Print the model summary
outputs.summary()
