# Loan Prediction Neural Network

A neural network implementation from scratch using NumPy to predict loan approval.

## How to Run the Project

### Setup

1. Create a virtual environment:

   ```
   python3 -m venv venv
   ```

2. Activate the virtual environment:

   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Model

Run the loan predictor with default parameters:

```
python loan_predictor.py
```

### Command-line Options

Customize the run with these parameters:

```
python loan_predictor.py --hidden1 12 --hidden2 8 --epochs 500 --lr 0.0005 --batch_size 16
```

Available options:

- `--data`: Path to the data file (default: 'loan-train.csv')
- `--hidden1`: Number of neurons in first hidden layer (default: 12)
- `--hidden2`: Number of neurons in second hidden layer (default: 8)
- `--epochs`: Number of training epochs (default: 500)
- `--lr`: Learning rate (default: 0.0005)
- `--batch_size`: Batch size for mini-batch gradient descent (default: 16)
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--output`: Path to save predictions (default: 'predictions.csv')
