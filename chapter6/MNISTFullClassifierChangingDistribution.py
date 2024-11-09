import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import fetch_openml


def ensure_plots_dir():
    """Create a plot folder if one doesn't exist"""
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir


def load_mnist():
    """
    - Loads MNIST dataset through scikit-learn
    
    - converts into float32 to save memory

    - normalizes the values of the pixel from 0-255 to 0-1

    - one hot encoding of the classes (so instead of having one integer going from 0 to 9, 
        we a binary vector of dimension 10)
    
    
    """
    print("Loading MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    X = X.astype('float32')
    
    X /= 255.0
    

    y = y.astype(int)
    y_onehot = np.zeros((y.shape[0], 10))
    y_onehot[np.arange(y.shape[0]), y] = 1
    
    return train_test_split(X, y_onehot, test_size=0.2, random_state=42)


class MNISTClassifierWithDist:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.01, 
                 init_type='he', distribution='normal'):
        """
        Initializes the neural net with different inizialitation strategies. 
        Biases are always set to zero during the initialization.
        
        Args:
            input_size (int): Dimension of the input (28*28 = 784 pixel for the MNIST)
            hidden_size (int): Number of neurons in the hidden layers
            output_size (int): Number of classes (in our case a 10-dim vector)
            learning_rate (float): Learning rate
            init_type (str): Initialization type ('he', 'xavier', 'lecun', 'simple')
            distribution (str): Distribution type for each initialization ('normal' o 'uniform')
        """
        self.lr = learning_rate
        
        if distribution == 'normal':
            if init_type == 'he':
                scale1 = np.sqrt(2.0/input_size)
                scale2 = np.sqrt(2.0/hidden_size)
                self.W1 = np.random.normal(0, scale1, (input_size, hidden_size))
                self.W2 = np.random.normal(0, scale2, (hidden_size, output_size))
                
            elif init_type == 'xavier':
                scale1 = np.sqrt(2.0/(input_size + hidden_size))
                scale2 = np.sqrt(2.0/(hidden_size + output_size))
                self.W1 = np.random.normal(0, scale1, (input_size, hidden_size))
                self.W2 = np.random.normal(0, scale2, (hidden_size, output_size))
                
            elif init_type == 'lecun':
                scale1 = np.sqrt(1.0/input_size)
                scale2 = np.sqrt(1.0/hidden_size)
                self.W1 = np.random.normal(0, scale1, (input_size, hidden_size))
                self.W2 = np.random.normal(0, scale2, (hidden_size, output_size))
                
            else:  # simple
                self.W1 = np.random.normal(0, 0.01, (input_size, hidden_size))
                self.W2 = np.random.normal(0, 0.01, (hidden_size, output_size))
                
        elif distribution == 'uniform':
            if init_type == 'he':
                limit1 = np.sqrt(6.0/input_size) 
                limit2 = np.sqrt(6.0/hidden_size)
                self.W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
                self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, output_size))
                
            elif init_type == 'xavier':
                limit1 = np.sqrt(6.0/(input_size + hidden_size))
                limit2 = np.sqrt(6.0/(hidden_size + output_size))
                self.W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
                self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, output_size))
                
            elif init_type == 'lecun':
                limit1 = np.sqrt(3.0/input_size)
                limit2 = np.sqrt(3.0/hidden_size)
                self.W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
                self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, output_size))
                
            else:  # simple
                self.W1 = np.random.uniform(-0.01, 0.01, (input_size, hidden_size))
                self.W2 = np.random.uniform(-0.01, 0.01, (hidden_size, output_size))
        

        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)
        

        self.init_type = init_type
        self.distribution = distribution
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input data (batch_size, 784)
            
        Returns:
            tuple: (neural net output, intermediate activations)
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.softmax(self.z2)
        return self.output
    
    def backward(self, X, y, output):
        """
        Backward propagation
        
        Args:
            X: Input data
            y: Target labels (one-hot encoded)
            output: neural net output
        """
        batch_size = X.shape[0]
        
        # Second layer gradients computation
        delta2 = output - y
        dW2 = np.dot(self.a1.T, delta2) / batch_size
        db2 = np.sum(delta2, axis=0) / batch_size
        
        # First layer gradients computation
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, delta1) / batch_size
        db1 = np.sum(delta1, axis=0) / batch_size
        
        # Weight update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        """predict the classes for the input data"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def train(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=128):
        """
        neural net training
        
        Args:
            X: Training data
            y: Target labels
            X_val: Validation data
            y_val: Validation labels
            epochs (int): Number of epochs
            batch_size (int): Batch dimension
        """
        n_samples = X.shape[0]
        training_loss = []
        validation_acc = []
        
        start_time = datetime.now()
        print(f"Training start: {start_time.strftime('%H:%M:%S')}")
        
        for epoch in range(epochs):
            epoch_start = datetime.now()
            
            # Shuffle 
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
            
            loss = 0
            for i in range(0, n_samples, batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                output = self.forward(batch_X)
                self.backward(batch_X, batch_y, output)
                
                loss += -np.mean(np.sum(batch_y * np.log(output + 1e-8), axis=1))
            
            loss /= (n_samples // batch_size)
            training_loss.append(loss)
            
            # Validation accuracy
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_acc = np.mean(val_pred == np.argmax(y_val, axis=1))
                validation_acc.append(val_acc)
                val_status = f", Val Acc: {val_acc:.4f}"
            else:
                val_status = ""
            
            # Stampa con timestamp
            current_time = datetime.now()
            epoch_duration = current_time - epoch_start
            print(f"[{current_time.strftime('%H:%M:%S')}] Epoch {epoch+1}/{epochs}, "
                  f"Loss: {loss:.4f}{val_status}, duration: {epoch_duration.total_seconds():.2f}s")
        
        end_time = datetime.now()
        total_duration = end_time - start_time
        print(f"\nTraining complete: {end_time.strftime('%H:%M:%S')}")
        print(f"total duration: {total_duration.total_seconds():.2f}s")
        
        return training_loss, validation_acc

def evaluate_model(model, X_test, y_test, plots_dir):
    """
    Evaluates the model on the test dataset
    """

    predictions = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    

    accuracy = np.mean(predictions == y_true)
    print("\n Evaluation results:")
    print(f"Accuracy on the test set: {accuracy:.4f}")
    

    print("\nReport:")
    print(classification_report(y_true, predictions))
    

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    confusion_filename = os.path.join(plots_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(confusion_filename)
    plt.close()
    print(f"\nCOnfusion matrix saved as: {confusion_filename}")
    
    errors = predictions != y_true
    error_indices = np.where(errors)[0]
    
    if len(error_indices) > 0:
        plt.figure(figsize=(15, 5))
        num_errors_to_show = min(5, len(error_indices))
        
        for i in range(num_errors_to_show):
            idx = error_indices[i]
            plt.subplot(1, 5, i+1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f'True: {y_true[idx]}\nPred: {predictions[idx]}')
            plt.axis('off')
        
        error_filename = os.path.join(plots_dir, f'error_examples_{timestamp}.png')
        plt.savefig(error_filename)
        plt.close()
        print(f"Error examples saved as: {error_filename}")

def save_training_plots(train_loss, val_acc, plots_dir):
    """Saves training graphs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    loss_filename = os.path.join(plots_dir, f'training_loss_{timestamp}.png')
    plt.savefig(loss_filename)
    plt.close()
    print(f"Loss graph saved as: {loss_filename}")
    
    # validation accuracy
    if val_acc:
        plt.figure(figsize=(10, 5))
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Validation Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        acc_filename = os.path.join(plots_dir, f'validation_accuracy_{timestamp}.png')
        plt.savefig(acc_filename)
        plt.close()
        print(f"Validation accuracy saved as: {acc_filename}")

def compare_distributions(X_train, y_train, X_val, y_val, plots_dir):
    """
    Compare different initializations (he, xavier, lecun, simple) with different statistical distribution (normal and uniform)
    """
    init_types = ['he', 'xavier', 'lecun', 'simple']
    distributions = ['normal', 'uniform']
    results = {}
    
    for init_type in init_types:
        for dist in distributions:
            name = f"{init_type}_{dist}"
            print(f"\nTraining with {init_type} initialization, distribution {dist}...")
            
            clf = MNISTClassifierWithDist(init_type=init_type, distribution=dist)
            loss_history, val_acc = clf.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=10)
            results[name] = {'loss': loss_history, 'val_acc': val_acc}
    
    # loss comparison plot
    plt.figure(figsize=(15, 7))
    for name, data in results.items():
        plt.plot(data['loss'], label=name)
    plt.title('Compare Loss for Different Initializations and Distributions')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(plots_dir, f'dist_comparison_loss_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    # validation accuracy comparison plot
    plt.figure(figsize=(15, 7))
    for name, data in results.items():
        plt.plot(data['val_acc'], label=name)
    plt.title('Validation Accuracy Different Initializations and Distributions')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    filename = os.path.join(plots_dir, f'dist_comparison_acc_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return results

def main():
    plots_dir = ensure_plots_dir()
    
    X_train, X_test, y_train, y_test = load_mnist()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Compare distributions
    results = compare_distributions(X_train, y_train, X_val, y_val, plots_dir)

if __name__ == "__main__":
    main()
