import numpy as np
import pandas as pd
from neural_network import NeuralNetwork

def main():
    # Load the dataset from CSV file
    print("Loading dataset from gender_data.csv...")
    df = pd.read_csv('gender_data.csv')
    
    # Get features and labels
    X = df[['height', 'weight']].values
    y = df[['gender']].values
    
    # Split data into training (80%) and testing (20%) sets
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * 0.8)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Dataset split into {len(X_train)} training samples and {len(X_test)} test samples")
    
    # Normalize inputs (subtract mean and divide by std)
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train_normalized = (X_train - X_mean) / X_std
    X_test_normalized = (X_test - X_mean) / X_std
    
    print("\nFeature statistics:")
    print(f"Mean: {X_mean}")
    print(f"Std: {X_std}")
    
    # Create a neural network with:
    # - 2 input neurons (height, weight)
    # - 4 neurons in first hidden layer (increased from 3)
    # - 4 neurons in second hidden layer (increased from 3)
    # - 1 output neuron (gender prediction)
    nn = NeuralNetwork(
        input_size=2,
        hidden1_size=4,
        hidden2_size=4,
        output_size=1,
        learning_rate=0.5
    )
    
    # Train the network
    print("\nTraining the neural network...")
    losses = nn.train(X_train_normalized, y_train, epochs=5000, print_every=500)
    
    # Test the network on test data
    print("\nEvaluating on test data:")
    predictions = nn.predict(X_test_normalized)
    
    # Calculate classification accuracy with threshold of 0.5
    predictions_binary = (predictions > 0.5).astype(int)
    accuracy = np.mean(predictions_binary == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Print some example predictions
    print("\nSample predictions:")
    for i in range(min(5, len(X_test))):
        print(f"Input: Height={X_test[i][0]:.1f}cm, Weight={X_test[i][1]:.1f}kg -> True: {y_test[i][0]}, Predicted: {predictions[i][0]:.4f}")
    
    # Test with a few custom examples
    print("\nCustom examples:")
    custom_samples = np.array([
        [190, 85],  # Tall and heavy (likely male)
        [160, 55],  # Short and light (likely female)
        [175, 65]   # Medium height and weight (could be either)
    ])
    
    custom_normalized = (custom_samples - X_mean) / X_std
    custom_predictions = nn.predict(custom_normalized)
    
    labels = ["Male", "Female"]
    for i, sample in enumerate(custom_samples):
        pred_value = custom_predictions[i][0]
        pred_label = labels[1] if pred_value > 0.5 else labels[0]
        print(f"Person with height={sample[0]}cm, weight={sample[1]}kg -> Prediction: {pred_value:.4f} ({pred_label})")

if __name__ == "__main__":
    main() 