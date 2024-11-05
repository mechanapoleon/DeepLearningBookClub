import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# load and preprocess the data 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# initialize the parameters
np.random.seed(1)
weights = np.random.randn(784, 10) * 0.01
bias = np.zeros((1, 10))

learning_rate = 0.1
epochs = 10

# training loop
for epoch in range(epochs):
    # forward pass
    logits = np.dot(X_train, weights) + bias
    predictions = softmax(logits)
    
    # cross-entropy loss
    loss = -np.mean(np.sum(y_train * np.log(predictions + 1e-8), axis=1))
    
    # backpropagation
    d_logits = (predictions - y_train) / predictions.shape[0]
    d_weights = np.dot(X_train.T, d_logits)
    d_bias = np.sum(d_logits, axis=0, keepdims=True)
    
    # update weight and bias
    weights -= learning_rate * d_weights
    bias -= learning_rate * d_bias
    
    # print the loss every 1 epoch
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# validation
logits_test = np.dot(X_test, weights) + bias
predictions_test = softmax(logits_test)
predicted_labels = np.argmax(predictions_test, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_labels == true_labels)
print(f"\nTest accuracy: {accuracy:.4f}")

# plot the first 15 test images with their predicted and true labels
plt.figure(figsize=(10, 6))

for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predizione: {predicted_labels[i]}, Reale: {true_labels[i]}")
    plt.axis('off')  # Nasconde gli assi per estetica

plt.tight_layout()  # Dispone meglio la griglia
plt.show()
