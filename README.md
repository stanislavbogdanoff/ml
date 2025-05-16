# Neural Network from Scratch

This project implements a feedforward neural network from scratch using only NumPy, without relying on any deep learning libraries like PyTorch or TensorFlow.

## Features

- A fully-connected feedforward neural network with two hidden layers
- Implemented using object-oriented design
- Manual implementation of forward propagation and backpropagation
- Sigmoid activation function for all layers
- Mean Squared Error (MSE) loss function
- Gradient descent optimizer
- Synthetic data generation for training and testing

## Architecture

- Input layer: 2 neurons (height and weight)
- First hidden layer: 4 neurons
- Second hidden layer: 4 neurons
- Output layer: 1 neuron (binary classification)

## Installation

1. Ensure you have Python 3 installed.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Generation

The project includes a data generation script that creates synthetic height/weight data with gender labels:

```bash
python generate_dataset.py
```

This will:

1. Generate a balanced dataset of 120 samples (60 males, 60 females)
2. Create realistic integer values for heights (150-200 cm) and weights (45-100 kg)
3. Save the data to `gender_data.csv`

The data generator uses normal distributions with different parameters for males and females:

- Males: Mean height 178cm, mean weight 80kg
- Females: Mean height 165cm, mean weight 62kg

## Usage

After generating the dataset, run the main.py file to train and test the neural network:

```bash
python main.py
```

This will:

1. Load the dataset from `gender_data.csv`
2. Split it into training (80%) and testing (20%) sets
3. Normalize the features by subtracting the mean and dividing by standard deviation
4. Train the neural network for 5000 epochs
5. Evaluate the model on test data and report accuracy
6. Test with custom examples to show generalization

## Customization

You can modify the neural network architecture by changing the parameters in main.py:

```python
nn = NeuralNetwork(
    input_size=2,
    hidden1_size=4,
    hidden2_size=4,
    output_size=1,
    learning_rate=0.5
)
```

You can also adjust the number of epochs and how often to print the loss:

```python
losses = nn.train(X_train_normalized, y_train, epochs=5000, print_every=500)
```

For data generation, you can modify parameters in generate_dataset.py:

```python
df = generate_gender_dataset(n_samples=120, random_seed=42)
```
