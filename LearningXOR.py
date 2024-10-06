import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(1)
weights_input_hidden = np.random.randn(2, 3)
bias_hidden = np.zeros((1, 3))
weights_hidden_output = np.random.randn(3, 1)
bias_output = np.zeros((1, 1))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

learning_rate = 0.1
epochs = 10000

print("Training started...")
print(f"Configuration: 2 input neurons, 3 hidden neurons (ReLU), 1 output neuron (Sigmoid)")
print(f"Learning rate: {learning_rate}, Epochs: {epochs}\n")

for epoch in range(epochs):
    hidden_layer = relu(np.dot(X, weights_input_hidden) + bias_hidden)
    output_layer = sigmoid(np.dot(hidden_layer, weights_hidden_output) + bias_output)
    
    loss = np.mean(np.square(y - output_layer))
    
    d_output = (output_layer - y) * sigmoid_derivative(output_layer)
    d_hidden = np.dot(d_output, weights_hidden_output.T) * relu_derivative(hidden_layer)
    
    weights_hidden_output -= learning_rate * np.dot(hidden_layer.T, d_output)
    bias_output -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
    weights_input_hidden -= learning_rate * np.dot(X.T, d_hidden)
    bias_hidden -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

print("\nTraining complete!")
print(f"Final loss: {loss:.4f}\n")

print("Results:")
print("Input (X) | Target (y) | Prediction")
print("-" * 35)
for i in range(len(X)):
    input_str = f"{X[i][0]} {X[i][1]}"
    target = y[i][0]
    prediction = output_layer[i][0]
    print(f"  {input_str}    |    {target}     |   {prediction:.4f}")

print("\nResult interpretation:")
print("- A prediction near 0 means that the NN thinks that it should be  0")
print("- Same for 1")
print("- The NN has learnt correctly if the output are near the target")