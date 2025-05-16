import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01):
        """
        Initialize a neural network with two hidden layers.
        
        Args:
            input_size: Number of input features
            hidden1_size: Number of neurons in the first hidden layer
            hidden2_size: Number of neurons in the second hidden layer
            output_size: Number of output neurons (1 for binary classification)
            learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier/Glorot initialization for better gradient flow
        # Weights for connections between input and first hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden1_size) * np.sqrt(1 / self.input_size)
        self.b1 = np.zeros((1, self.hidden1_size))
        
        # Weights for connections between first and second hidden layer
        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(1 / self.hidden1_size)
        self.b2 = np.zeros((1, self.hidden2_size))
        
        # Weights for connections between second hidden layer and output
        self.W3 = np.random.randn(self.hidden2_size, self.output_size) * np.sqrt(1 / self.hidden2_size)
        self.b3 = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        """
        Sigmoid activation function: 1 / (1 + exp(-x))
        Clip values to avoid overflow/underflow
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of sigmoid function: sigmoid(x) * (1 - sigmoid(x))
        """
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Args:
            X: Input features (batch_size, input_size)
            
        Returns:
            Outputs of each layer for backpropagation
        """
        # Linear transformation and activation for first hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Linear transformation and activation for second hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        # Linear transformation and activation for output layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute Mean Squared Error loss
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            MSE loss value
        """
        return np.mean(np.square(y_true - y_pred))
    
    def backward(self, X, y):
        """
        Backward pass to update weights using gradient descent
        
        Args:
            X: Input features
            y: True labels
        """
        m = X.shape[0]  # Number of examples
        
        # Output layer error
        dz3 = self.a3 - y
        dW3 = (1/m) * np.dot(self.a2.T, dz3)
        db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)
        
        # Second hidden layer error
        dz2 = np.dot(dz3, self.W3.T) * self.sigmoid_derivative(self.z2)
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # First hidden layer error
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs, print_every=10):
        """
        Train the neural network
        
        Args:
            X: Input features
            y: True labels
            epochs: Number of training iterations
            print_every: How often to print the loss
            
        Returns:
            List of losses during training
        """
        losses = []
        
        for i in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y)
            
            # Print loss every print_every epochs
            if i % print_every == 0:
                print(f"Epoch {i}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions with the trained network
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.forward(X) 