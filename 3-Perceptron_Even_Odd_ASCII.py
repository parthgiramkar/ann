# 3: Program using Perceptron Neural Network to recognize even and
# odd numbers. Given numbers are in ASCII from 0 to 9

import numpy as np # Import NumPy for matrix operations and array handling

# --- Define the Perceptron Class ---
class Perceptron:
    # Constructor to initialize the model
    def __init__(self, input_size, lr=0.1):
        # Initialize weights with zeros. We add +1 to the input size for the 'Bias'
        self.W = np.zeros(input_size + 1)
        # Learning Rate (lr) determines how much we adjust weights during training
        self.lr = lr

    # Activation Function: Step function (returns 1 if x >= 0, else 0)
    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    # Prediction Method: Predicts the output for a given input
    def predict(self, x):
        # Insert a '1' at the start of the input to account for the Bias weight
        x = np.insert(x, 0, 1)
        # Calculate the Dot Product: (Weights * Inputs) + Bias
        z = self.W.T.dot(x)
        # Pass the result through the activation function
        a = self.activation_fn(z)
        return a

    # Training Method: Adjusts weights based on errors
    def train(self, X, Y, epochs):
        for _ in range(epochs):
            for i in range(Y.shape[0]):
                x = X[i]
                # Step 1: Make a prediction
                y_pred = self.predict(x)
                # Step 2: Calculate the error (Target - Prediction)
                error = Y[i] - y_pred
                # Step 3: Update weights if there was an error
                # Rule: New Weight = Old Weight + (Learning Rate * Error * Input)
                self.W = self.W + self.lr * error * np.insert(x, 0, 1)

# --- Define the Training Data (Digits 0-9 in One-Hot encoding) ---
# Each row represents a digit (0 to 9) using 10 nodes
X = np.array([
    [0,0,0,0,0,0,1,0,0,0], # Representing 0
    [0,0,0,0,0,0,0,1,0,0], # Representing 1
    [0,0,0,0,0,0,0,0,1,0], # Representing 2
    [0,0,0,0,0,0,0,0,0,1], # Representing 3
    [0,0,0,0,0,0,1,1,0,0], # Representing 4
    [0,0,0,0,0,1,0,1,0,0], # Representing 5
    [0,0,0,0,0,1,1,1,0,0], # Representing 6
    [0,0,0,0,0,1,1,1,1,0], # Representing 7
    [0,0,0,0,0,1,0,1,1,1], # Representing 8
    [0,0,0,0,0,0,1,1,1,1]  # Representing 9
])

# Labels: 0 for Even, 1 for Odd
Y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# --- Create and Train the Perceptron ---
# input_size=10 because each input has 10 features
perceptron = Perceptron(input_size=10)
# Train the model for 100 iterations (epochs)
perceptron.train(X, Y, epochs=100)

# --- Test the trained model ---
print("--- Perceptron Even/Odd Test Results ---")
for i in range(X.shape[0]):
    x = X[i]
    prediction = perceptron.predict(x)
    result = "Even" if prediction == 0 else "Odd"
    print(f"Input Data {i}: Recognized as -> {result}")




















"""
--- Code Explanation ---
This script uses a Single-Layer Perceptron to learn the pattern of Even and Odd numbers.

1. Perceptron Class: This is our neural network 'blueprint'.
2. Initialization: We create weights (W) and set a learning rate (lr).
3. Insert(x, 0, 1): We add a '1' to every input. This '1' is multiplied by the first weight, 
    which acts as the 'Bias'.
4. Training Loop: The network looks at the data, makes a guess (predict), and if it's wrong, 
    it tweaks the weights using the 'Perceptron Learning Rule'.
5. Dataset: We used a simple binary-like representation for numbers 0-9.

---------------------------------------------------------------------------------------
                          THEORY NOTES FOR VIVA (EXTENDED)
---------------------------------------------------------------------------------------

1. WHAT IS A PERCEPTRON? (As per Lab Manual)
   - Developed by Frank Rosenblatt in the late 1950s.
   - It is a supervised learning algorithm for binary classifiers.
   - It is a single-layer neural network that mimics how a biological neuron works.
   - The output is decided based on a single activation function (threshold).

2. TYPES OF PERCEPTRON MODELS:
   - Single Layer Perceptron: Can only learn 'Linearly Separable' patterns (like 
     Even/Odd or AND/OR). It has no hidden layers.
   - Multi-Layered Perceptron (MLP): Contains one or more hidden layers. It can 
     solve complex non-linear problems like XOR.

3. THE TWO STAGES OF LEARNING:
   - Forward Stage: Input data is multiplied by weights, a bias is added, and the 
     activation function produces an output.
   - Backward Stage: The model calculates the error (Target - Prediction) and 
     modifies the weight and bias values to reduce that error.

4. THE LEARNING RULE (Mathematical Formulas):
   - Weight Update: W(new) = W(old) + α * (y - y_hat) * x
   - Bias Update: b(new) = b(old) + α * (y - y_hat)
   - (Where α is the Learning Rate, y is Target, and y_hat is Prediction).

5. ADVANTAGES OF PERCEPTRON:
   - Speed: Once trained, it provides extremely quick predictions.
   - Consistency: It maintains the same accuracy ratio for both small and large 
     datasets if the data is linearly separable.

6. LIMITATIONS:
   - It cannot solve non-linear problems (e.g., XOR).
   - If the data points are not clearly separable by a line, the algorithm will 
     never converge (it will keep looping).

---------------------------------------------------------------------------------------
                             EXPANDED VIVA Q&A
---------------------------------------------------------------------------------------

Q1: Who originally developed the Perceptron Learning Algorithm?
A1: Frank Rosenblatt in the late 1950s.

Q2: What are the primary components of a Perceptron?
A2: Input nodes (Features), Weights (Importance), Bias (Threshold shift), 
    Summation function, and Activation function.

Q3: Why do we use a "Learning Rate" (α)?
A3: It controls how much we adjust the weights in each step. If it's too high, 
    we might miss the optimal solution; if too low, training will be very slow.

Q4: What is the "Step Function" in your code?
A4: It is the activation function. It acts as a hard threshold: if the weighted 
    sum is positive, it returns 1; otherwise, it returns 0.

Q5: In this practical, what does the input array represent?
A5: It is an "ASCII-like" binary representation of digits 0-9. Each bit represents 
    a specific feature of the number that helps the model distinguish Even from Odd.

Q6: What does "Linear Separability" mean?
A6: It means you can draw a single straight line on a graph to completely 
    separate the "Even" dots from the "Odd" dots.

Q7: What is the difference between a Perceptron and a Neural Network?
A7: A Perceptron is the simplest form of a Neural Network (a single neuron). 
    A modern "Neural Network" usually refers to multiple neurons arranged 
    in multiple layers (Deep Learning).

Q8: What happens to the error as training progresses?
A8: In a successful training process, the error should decrease with each epoch 
    until it reaches zero (perfect classification).

---------------------------------------------------------------------------------------
"""
