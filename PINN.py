import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the physics-based constraints (example: 1D heat conduction equation)
def physics_loss(model, x_phys):
    u_pred = model(x_phys)
    # Example physics-based constraint: Laplace(u) = 0
    u_x = tf.gradients(u_pred, x_phys)[0]
    u_xx = tf.gradients(u_x, x_phys)[0]
    physics_loss = tf.reduce_mean(tf.square(u_xx))
    return physics_loss

# Define the data-driven loss
def data_loss(model, x_data, y_data):
    y_pred = model(x_data)
    data_loss = tf.reduce_mean(tf.square(y_pred - y_data))
    return data_loss

# Define the PINN model
class PINNModel(keras.Model):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.hidden1 = layers.Dense(50, activation='tanh', input_dim=1)
        self.hidden2 = layers.Dense(50, activation='tanh')
        self.output_layer = layers.Dense(1, activation='linear')

    def call(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.output_layer(x)

# Generate training data (example: 1D heat conduction)
x_data = np.random.uniform(-1, 1, (100, 1))  # Random training points
y_data = np.sin(np.pi * x_data)  # Example function

# Generate physics-based training data
x_phys = np.linspace(-1, 1, 100).reshape(-1, 1)

# Create PINN model
model = PINNModel()

# Define optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# Training loop
epochs = 10000
for epoch in range(epochs):
    # Compute total loss
    with tf.GradientTape(persistent=True) as tape:
        physics_loss_value = physics_loss(model, x_phys)
        data_loss_value = data_loss(model, x_data, y_data)
        total_loss = data_loss_value + 1.0 * physics_loss_value  # Adjust the weight for physics_loss

    # Compute gradients and update model parameters
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss.numpy()}, Data Loss: {data_loss_value.numpy()}, Physics Loss: {physics_loss_value.numpy()}")

# After training, use the model for predictions
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
y_pred = model(x_test)

# Plot results
import matplotlib.pyplot as plt

plt.scatter(x_data, y_data, label='Training Data')
plt.plot(x_test, y_pred, label='PINN Prediction', color='r')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
