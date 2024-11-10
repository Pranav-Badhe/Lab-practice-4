#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Import required libraries
import tensorflow as tf  # Core library for building and training models
from tensorflow.keras import layers, models  # Layers and models to define the autoencoder architecture
import numpy as np  # For data manipulation and array handling
import matplotlib.pyplot as plt  # For visualizing the data and results
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets

# Step 2: Load and preprocess the dataset
# Here, we'll use the MNIST dataset for simplicity, simulating anomalies by focusing on one digit, e.g., "1" as normal
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data by scaling pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Select only the "normal" class, e.g., digit "1" for training (simulating anomalies for other digits)
normal_class = 1
x_train_normal = x_train[y_train == normal_class]
x_test_normal = x_test[y_test == normal_class]

# Flatten the images to 1D vectors for the autoencoder
x_train_normal = x_train_normal.reshape(-1, 28 * 28)
x_test_normal = x_test_normal.reshape(-1, 28 * 28)

# Step 3: Build the encoder part of the autoencoder
input_dim = x_train_normal.shape[1]  # Input dimension (784 for flattened 28x28 images)
latent_dim = 64  # Dimension of the latent space representation

encoder = models.Sequential([
    layers.Input(shape=(input_dim,)),  # Input layer, accepting 784-dimensional vectors
    layers.Dense(128, activation="relu"),  # First dense layer with 128 neurons and ReLU activation
    layers.Dense(64, activation="relu"),  # Second dense layer with 64 neurons
    layers.Dense(latent_dim, activation="relu")  # Latent layer with 64 neurons for the compressed representation
])

# Step 4: Build the decoder part of the autoencoder
decoder = models.Sequential([
    layers.Input(shape=(latent_dim,)),  # Input layer matching latent_dim
    layers.Dense(64, activation="relu"),  # First dense layer to expand from latent representation
    layers.Dense(128, activation="relu"),  # Second dense layer
    layers.Dense(input_dim, activation="sigmoid")  # Output layer reconstructing the 784-dimensional input
])

# Step 5: Connect encoder and decoder to form the autoencoder model
autoencoder = models.Sequential([encoder, decoder])  # Autoencoder combines encoder and decoder

# Step 6: Compile the model with optimizer, loss, and metrics
autoencoder.compile(optimizer="adam",  # Optimizer to update weights
                    loss="mse",  # Mean Squared Error as loss to measure reconstruction quality
                    metrics=["mae"])  # Mean Absolute Error as an additional metric

# Step 7: Train the model on normal data only (anomaly detection approach)
history = autoencoder.fit(
    x_train_normal, x_train_normal,  # Both input and output are the same for autoencoders
    epochs=50,  # Number of epochs
    batch_size=256,  # Batch size
    validation_data=(x_test_normal, x_test_normal)  # Validation on normal test data
)

# Step 8: Evaluate and visualize the model's performance
# Plot training & validation loss for each epoch
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')  # Plot training loss
plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot validation loss
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Step 9: Detect anomalies by setting a reconstruction error threshold
# Calculate reconstruction error on normal test data
reconstructed_normal = autoencoder.predict(x_test_normal)
mse_normal = np.mean(np.power(x_test_normal - reconstructed_normal, 2), axis=1)

# Set threshold for anomaly detection (95th percentile of reconstruction errors)
threshold = np.percentile(mse_normal, 95)
print(f"Reconstruction error threshold: {threshold}")

# Test the autoencoder with a mix of normal and anomalous data (using digit "7" as anomalies)
x_test_anomalous = x_test[y_test == 7].reshape(-1, 28 * 28)  # Reshape anomalous images
reconstructed_anomalous = autoencoder.predict(x_test_anomalous)  # Reconstruct anomalous images
mse_anomalous = np.mean(np.power(x_test_anomalous - reconstructed_anomalous, 2), axis=1)

# Plot histograms of reconstruction errors for normal vs. anomalous samples
plt.figure(figsize=(10, 5))
plt.hist(mse_normal, bins=50, alpha=0.6, label="Normal")
plt.hist(mse_anomalous, bins=50, alpha=0.6, label="Anomalous")
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel("Reconstruction error")
plt.ylabel("Number of samples")
plt.legend()
plt.title("Reconstruction Error for Normal and Anomalous Data")
plt.show()

# Step 10: Predict anomalies based on the threshold
# Classify samples as anomalies if their reconstruction error exceeds the threshold
anomalies = mse_anomalous > threshold
print("Number of detected anomalies:", np.sum(anomalies))


# In[ ]:




