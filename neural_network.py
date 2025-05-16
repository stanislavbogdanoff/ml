import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.001, reg_lambda=0.001):
        """
        Initialize a neural network with two hidden layers
        
        Parameters:
        input_size (int): Number of input features
        hidden1_size (int): Number of neurons in the first hidden layer
        hidden2_size (int): Number of neurons in the second hidden layer
        output_size (int): Number of output neurons (1 for binary classification)
        learning_rate (float): Learning rate for gradient descent
        reg_lambda (float): L2 regularization strength
        """
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # He initialization for ReLU activations in hidden layers
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2. / input_size) 
        self.b1 = np.zeros((1, hidden1_size))
        
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2. / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        
        # Glorot/Xavier initialization for sigmoid output
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(1. / hidden2_size)
        self.b3 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        """Sigmoid activation function with safe clipping"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU function"""
        return (z > 0).astype(float)
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network
        
        Parameters:
        X (numpy.ndarray): Input features of shape (batch_size, input_size)
        
        Returns:
        tuple: Final output after forward pass
        """
        # First hidden layer with ReLU
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Second hidden layer with ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        # Output layer with sigmoid
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss with L2 regularization
        
        Parameters:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted values
        
        Returns:
        float: Binary cross-entropy loss with regularization
        """
        m = y_true.shape[0]
        
        # Apply class weights to handle imbalance
        # Weight positive samples higher than negative samples
        pos_weight = 3.0  # Higher weight for positive class
        neg_weight = 1.0
        
        weights = np.ones_like(y_true)
        weights[y_true == 1] = pos_weight
        weights[y_true == 0] = neg_weight
        
        # Binary cross-entropy loss with weights
        epsilon = 1e-15  # Small constant to avoid log(0)
        loss = -np.mean(
            weights * (
                y_true * np.log(np.clip(y_pred, epsilon, 1.0)) + 
                (1 - y_true) * np.log(np.clip(1 - y_pred, epsilon, 1.0))
            )
        )
        
        # L2 regularization term
        reg_term = (self.reg_lambda / (2 * m)) * (
            np.sum(np.square(self.W1)) + 
            np.sum(np.square(self.W2)) + 
            np.sum(np.square(self.W3))
        )
        
        return loss + reg_term
    
    def backward_propagation(self, X, y):
        """
        Perform backward propagation to compute gradients
        
        Parameters:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): True labels
        
        Returns:
        tuple: Gradients for all weights and biases
        """
        m = X.shape[0]
        
        # Apply fixed class weights
        pos_weight = 3.0  # Higher weight for positive class
        neg_weight = 1.0
        
        weights = np.ones_like(y)
        weights[y == 1] = pos_weight
        weights[y == 0] = neg_weight
        
        # Output layer error (binary cross-entropy gradient)
        dz3 = weights * (self.a3 - y)
        dW3 = (1/m) * np.dot(self.a2.T, dz3) + (self.reg_lambda/m) * self.W3
        db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)
        
        # Second hidden layer error with ReLU
        dz2 = np.dot(dz3, self.W3.T) * self.relu_derivative(self.z2)
        dW2 = (1/m) * np.dot(self.a1.T, dz2) + (self.reg_lambda/m) * self.W2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # First hidden layer error with ReLU
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1) + (self.reg_lambda/m) * self.W1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2, dW3, db3
    
    def update_parameters(self, dW1, db1, dW2, db2, dW3, db3):
        """
        Update network parameters using gradients from backpropagation
        
        Parameters:
        dW1, db1, dW2, db2, dW3, db3: Gradients for weights and biases
        """
        # Update weights and biases using gradient descent
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
    
    def train(self, X, y, epochs, batch_size=32, print_every=10):
        """
        Train the neural network using mini-batch gradient descent
        
        Parameters:
        X (numpy.ndarray): Training features
        y (numpy.ndarray): Training labels
        epochs (int): Number of training epochs
        batch_size (int): Size of mini-batches
        print_every (int): Print loss every this many epochs
        
        Returns:
        list: Training loss history
        """
        m = X.shape[0]
        loss_history = []
        best_loss = float('inf')
        best_weights = None
        patience = 20  # Early stopping patience
        no_improvement = 0
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, m, batch_size):
                end = min(i + batch_size, m)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                # Forward propagation
                y_pred = self.forward_propagation(X_batch)
                
                # Backward propagation
                dW1, db1, dW2, db2, dW3, db3 = self.backward_propagation(X_batch, y_batch)
                
                # Update parameters
                self.update_parameters(dW1, db1, dW2, db2, dW3, db3)
            
            # Calculate and print loss every print_every epochs
            if epoch % print_every == 0 or epoch == epochs - 1:
                y_pred = self.forward_propagation(X)
                loss = self.compute_loss(y, y_pred)
                loss_history.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
                
                # Check for improvement for early stopping
                if loss < best_loss:
                    best_loss = loss
                    # Save best weights
                    best_weights = {
                        'W1': self.W1.copy(), 'b1': self.b1.copy(),
                        'W2': self.W2.copy(), 'b2': self.b2.copy(),
                        'W3': self.W3.copy(), 'b3': self.b3.copy()
                    }
                    no_improvement = 0
                else:
                    no_improvement += print_every
                
                # Early stopping check
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Restore best weights if early stopping was triggered
        if best_weights is not None and no_improvement >= patience:
            self.W1 = best_weights['W1']
            self.b1 = best_weights['b1']
            self.W2 = best_weights['W2']
            self.b2 = best_weights['b2']
            self.W3 = best_weights['W3']
            self.b3 = best_weights['b3']
        
        return loss_history
    
    def predict(self, X):
        """
        Make predictions for input data
        
        Parameters:
        X (numpy.ndarray): Input features
        
        Returns:
        numpy.ndarray: Predicted values
        """
        return self.forward_propagation(X) 