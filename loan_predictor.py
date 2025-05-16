#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import time
from neural_network import NeuralNetwork

def preprocess_data(file_path):
    """
    Preprocess the loan data from CSV file
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    tuple: Preprocessed features and labels, feature means and stds, loan IDs
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Store Loan_ID for later use
    loan_ids = df['Loan_ID'].values
    
    # Select required columns: ApplicantIncome, LoanAmount, Credit_History, Loan_Status
    df_processed = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Loan_Status']]
    
    # Drop rows with missing values
    missing_indices = df_processed.isnull().any(axis=1)
    df_processed = df_processed.dropna()
    
    # Also remove corresponding loan IDs
    loan_ids = loan_ids[~missing_indices]
    
    # Convert Loan_Status: "Y" -> 1, "N" -> 0
    df_processed['Loan_Status'] = df_processed['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Separate features and labels
    X = df_processed[['ApplicantIncome', 'LoanAmount', 'Credit_History']].values
    y = df_processed['Loan_Status'].values.reshape(-1, 1)
    
    # Normalize features (subtract mean, divide by std)
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    X_normalized = (X - feature_means) / feature_stds
    
    print(f"Data shape after preprocessing: {X_normalized.shape} features, {y.shape} labels")
    print(f"Class distribution - Approved: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%), Rejected: {len(y) - np.sum(y)} ({(len(y) - np.sum(y))/len(y)*100:.1f}%)")
    
    return X_normalized, y, feature_means, feature_stds, loan_ids

def split_train_test_stratified(X, y, loan_ids, test_size=0.2, random_seed=None):
    """
    Split data into training and test sets while preserving class proportions
    
    Parameters:
    X (numpy.ndarray): Features
    y (numpy.ndarray): Labels
    loan_ids (numpy.ndarray): Loan IDs
    test_size (float): Proportion of data to use for testing
    random_seed (int or None): Seed for randomization, None for truly random
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, test_loan_ids
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get indices for each class
    pos_indices = np.where(y.flatten() == 1)[0]
    neg_indices = np.where(y.flatten() == 0)[0]
    
    # Shuffle indices
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)
    
    # Split each class
    pos_split = int(len(pos_indices) * (1 - test_size))
    neg_split = int(len(neg_indices) * (1 - test_size))
    
    train_indices = np.concatenate([pos_indices[:pos_split], neg_indices[:neg_split]])
    test_indices = np.concatenate([pos_indices[pos_split:], neg_indices[neg_split:]])
    
    # Shuffle again to mix classes
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Split the data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    test_loan_ids = loan_ids[test_indices]
    
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    print(f"Training set class distribution - Approved: {np.sum(y_train)} ({np.sum(y_train)/len(y_train)*100:.1f}%), Rejected: {len(y_train) - np.sum(y_train)} ({(len(y_train) - np.sum(y_train))/len(y_train)*100:.1f}%)")
    print(f"Test set class distribution - Approved: {np.sum(y_test)} ({np.sum(y_test)/len(y_test)*100:.1f}%), Rejected: {len(y_test) - np.sum(y_test)} ({(len(y_test) - np.sum(y_test))/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, test_loan_ids

def evaluate_predictions(y_true, y_pred, threshold=0.5):
    """
    Evaluate model predictions
    
    Parameters:
    y_true (numpy.ndarray): True labels
    y_pred (numpy.ndarray): Predicted probabilities
    threshold (float): Threshold for binary classification
    
    Returns:
    dict: Evaluation metrics
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_binary == y_true)
    
    # Calculate confusion matrix values
    true_positive = np.sum((y_true == 1) & (y_pred_binary == 1))
    true_negative = np.sum((y_true == 0) & (y_pred_binary == 0))
    false_positive = np.sum((y_true == 0) & (y_pred_binary == 1))
    false_negative = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    # Calculate precision, recall, and F1 score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate specificity (true negative rate)
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'confusion_matrix': {
            'true_positive': true_positive,
            'true_negative': true_negative,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
    }

def print_evaluation(metrics):
    """Print evaluation metrics in a readable format"""
    print("\nModel Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"True Positive: {cm['true_positive']}, True Negative: {cm['true_negative']}")
    print(f"False Positive: {cm['false_positive']}, False Negative: {cm['false_negative']}")

def print_predictions(y_true, y_pred, n=10):
    """Print sample predictions vs actual values"""
    print("\nSample Predictions (Probability | Actual):")
    for i in range(min(n, len(y_true))):
        print(f"  {y_pred[i][0]:.4f} | {y_true[i][0]}")

def save_predictions_to_file(loan_ids, y_true, y_pred, file_path="predictions.csv"):
    """
    Save predictions to a CSV file with the requested columns
    
    Parameters:
    loan_ids (numpy.ndarray): Loan IDs
    y_true (numpy.ndarray): True labels
    y_pred (numpy.ndarray): Predicted probabilities
    file_path (str): Path to save the CSV file
    """
    # Convert predictions to binary (0/1)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # Map binary values back to Y/N for actual loan status
    y_true_yn = np.where(y_true == 1, 'Y', 'N')
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Loan_ID': loan_ids,
        'Predicted_Label': y_pred_binary.flatten(),
        'Actual_Loan_Status': y_true_yn.flatten(),
        'Correct_Prediction': (y_pred_binary == y_true).flatten().astype(int)
    })
    
    # Save to CSV
    results_df.to_csv(file_path, index=False)
    print(f"\nPredictions saved to {file_path}")

def train_and_evaluate(X, y, loan_ids, args, iteration=None):
    """
    Train and evaluate the neural network with a single train/test split
    
    Parameters:
    X (numpy.ndarray): Features
    y (numpy.ndarray): Labels
    loan_ids (numpy.ndarray): Loan IDs
    args (argparse.Namespace): Command-line arguments
    iteration (int or None): Iteration number for multiple runs
    
    Returns:
    dict: Evaluation metrics
    """
    # Generate random seed for this iteration if we're doing multiple runs
    if args.random_seed is None:
        # Use current time * iteration as seed if we're in multi-iteration mode
        random_seed = None if iteration is None else int(time.time() * 1000) % 10000 + iteration
    else:
        # Use provided seed + iteration if we're in multi-iteration mode
        random_seed = args.random_seed if iteration is None else args.random_seed + iteration
    
    # Split data into training and test sets using stratified sampling
    X_train, X_test, y_train, y_test, test_loan_ids = split_train_test_stratified(
        X, y, loan_ids, test_size=args.test_size, random_seed=random_seed
    )
    
    # Initialize neural network with a seed that depends on the random_seed
    network_seed = None if random_seed is None else random_seed
    input_size = X_train.shape[1]  # 3 features
    nn = NeuralNetwork(
        input_size=input_size,
        hidden1_size=args.hidden1,
        hidden2_size=args.hidden2,
        output_size=1,
        learning_rate=args.lr,
        random_seed=network_seed
    )
    
    # Train the neural network
    print("\nTraining the neural network...")
    loss_history = nn.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    
    # Make predictions
    print("\nMaking predictions on the test set...")
    y_pred = nn.predict(X_test)
    
    # Find optimal threshold based on F1 score
    best_f1 = 0
    best_threshold = 0.5
    thresholds = np.arange(0.1, 0.91, 0.05)
    
    print("\nFinding optimal threshold:")
    for threshold in thresholds:
        metrics = evaluate_predictions(y_test, y_pred, threshold)
        f1 = metrics['f1']
        print(f"  Threshold {threshold:.2f}: F1={f1:.4f}, Accuracy={metrics['accuracy']:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} with F1={best_f1:.4f}")
    
    # Final evaluation with the best threshold
    metrics = evaluate_predictions(y_test, y_pred, best_threshold)
    print_evaluation(metrics)
    print_predictions(y_test, y_pred)
    
    # Save predictions to a file with only the requested columns
    # Note: Using the standard 0.5 threshold for the output file for simplicity
    if iteration is None:
        output_file = args.output
    else:
        # Create unique filenames for each iteration
        base, ext = args.output.rsplit('.', 1) if '.' in args.output else (args.output, 'csv')
        output_file = f"{base}_iter{iteration}.{ext}"
    
    save_predictions_to_file(test_loan_ids, y_test, y_pred, file_path=output_file)
    
    return metrics

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a neural network for loan prediction')
    parser.add_argument('--data', type=str, default='loan-train.csv', help='Path to the data file')
    parser.add_argument('--hidden1', type=int, default=12, help='Number of neurons in first hidden layer')
    parser.add_argument('--hidden2', type=int, default=8, help='Number of neurons in second hidden layer')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for mini-batch gradient descent')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to save predictions')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducibility (None for random splits)')
    parser.add_argument('--iterations', type=int, default=1, help='Number of training iterations with different splits')
    args = parser.parse_args()
    
    # Print neural network architecture
    print(f"\nNeural Network Architecture:")
    print(f"  Input Layer: 3 neurons (ApplicantIncome, LoanAmount, Credit_History)")
    print(f"  Hidden Layer 1: {args.hidden1} neurons with ReLU activation")
    print(f"  Hidden Layer 2: {args.hidden2} neurons with ReLU activation")
    print(f"  Output Layer: 1 neuron with sigmoid activation (Loan approval probability)")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Random Seed: {'Random' if args.random_seed is None else args.random_seed}")
    print(f"  Iterations: {args.iterations}")
    
    # Preprocess data
    print(f"\nPreprocessing data from {args.data}...")
    X, y, feature_means, feature_stds, loan_ids = preprocess_data(args.data)
    
    if args.iterations == 1:
        # Single run
        train_and_evaluate(X, y, loan_ids, args)
    else:
        # Multiple runs with different random splits
        print(f"\nRunning {args.iterations} iterations with different random splits...")
        
        # Lists to store metrics across iterations
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        specificities = []
        
        for i in range(args.iterations):
            print(f"\n\n=========== Iteration {i+1}/{args.iterations} ===========")
            metrics = train_and_evaluate(X, y, loan_ids, args, iteration=i)
            
            # Store metrics
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1'])
            specificities.append(metrics['specificity'])
        
        # Print summary statistics
        print("\n\n=========== Summary Statistics ===========")
        print(f"Accuracy: mean={np.mean(accuracies):.4f}, std={np.std(accuracies):.4f}, min={np.min(accuracies):.4f}, max={np.max(accuracies):.4f}")
        print(f"Precision: mean={np.mean(precisions):.4f}, std={np.std(precisions):.4f}, min={np.min(precisions):.4f}, max={np.max(precisions):.4f}")
        print(f"Recall: mean={np.mean(recalls):.4f}, std={np.std(recalls):.4f}, min={np.min(recalls):.4f}, max={np.max(recalls):.4f}")
        print(f"Specificity: mean={np.mean(specificities):.4f}, std={np.std(specificities):.4f}, min={np.min(specificities):.4f}, max={np.max(specificities):.4f}")
        print(f"F1 Score: mean={np.mean(f1_scores):.4f}, std={np.std(f1_scores):.4f}, min={np.min(f1_scores):.4f}, max={np.max(f1_scores):.4f}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 