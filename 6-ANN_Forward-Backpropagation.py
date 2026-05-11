# 6: Implement Artificial Neural Network training process in Python by using Forward Propagation, Back Propagation.

import numpy as np # Import NumPy for matrix operations and random weight generation

# --- Define the Neural Network Class ---
class NeuralNetwork:
    # 1. Initialize the network with Input, Hidden, and Output layers
    def __init__(self, input_size, hidden_size, output_size):
        # Weights (W1: Input to Hidden | W2: Hidden to Output)
        # We use random initialization because starting with zeros prevents learning
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        
        # Biases (b1: Hidden layer | b2: Output layer)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    # 2. Activation Function (Sigmoid): Squashes values between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 3. Sigmoid Derivative: Used to calculate the gradient during Backpropagation
    def sigmoid_derivative(self, x):
        # Formula: sigmoid(x) * (1 - sigmoid(x))
        return x * (1 - x)

    # 4. Forward Propagation: Moving the data from Input to Output
    def forward_propagation(self, X):
        # Layer 1 (Hidden): dot product + bias, then apply activation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2 (Output): dot product + bias, then apply activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.sigmoid(self.z2)
        
        return self.y_hat

    # 5. Backward Propagation: Moving the ERROR from Output back to Input to adjust weights
    def backward_propagation(self, X, y, y_hat, lr):
        # Step A: Calculate Output Layer Error
        self.error = y - y_hat
        # Calculate Delta (Error * Derivative)
        self.delta2 = self.error * self.sigmoid_derivative(y_hat)
        
        # Step B: Calculate Hidden Layer Error (sending delta2 back through W2)
        self.a1_error = self.delta2.dot(self.W2.T)
        self.delta1 = self.a1_error * self.sigmoid_derivative(self.a1)
        
        # Step C: Update Weights and Biases using the gradients
        # New Weight = Old Weight + (Inputs * Delta)
        self.W2 += self.a1.T.dot(self.delta2) * lr
        self.b2 += np.sum(self.delta2, axis=0, keepdims=True) * lr
        self.W1 += X.T.dot(self.delta1) * lr
        self.b1 += np.sum(self.delta1, axis=0) * lr

    # 6. Training Function: Running Forward and Backward passes for many iterations
    def train(self, X, y, epochs, lr):
        for i in range(epochs):
            y_hat = self.forward_propagation(X)
            self.backward_propagation(X, y, y_hat, lr)
            
            # Print the error every 1000 epochs to track progress
            if i % 1000 == 0:
                loss = np.mean(np.square(y - y_hat)) # Mean Squared Error
                print(f"Epoch {i} | Loss: {loss:.6f}")

# --- Step 7: Define XOR Data ---
# XOR cannot be solved by a simple Perceptron (Needs a hidden layer)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# --- Step 8: Execution ---

# Create a network: 2 Inputs -> 4 Hidden Neurons -> 1 Output
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Train the network
print("--- Starting Training (XOR Problem) ---")
nn.train(X, y, epochs=10000, lr=0.1)

# Final Predictions
print("\n--- Final Predictions after 10,000 Epochs ---")
predictions = nn.forward_propagation(X)
for i in range(len(X)):
    print(f"Input: {X[i]} | Predicted: {predictions[i][0]:.4f} | Target: {y[i][0]}")

























"""
--- Code Explanation ---
This script implements a Multi-Layer Perceptron (MLP) to solve the XOR logic gate.

1. Architecture: 2 Inputs -> 1 Hidden Layer (4 neurons) -> 1 Output.
2. Random Weights: We use `np.random.randn` because if all weights are 0, the neurons will all learn the exact same thing (Symmetry problem).
3. Forward Prop: We calculate the weighted sum and apply Sigmoid at each layer to move the signal forward.
4. Backward Prop (The "Learning" Part):
   - We calculate the error at the output.
   - We use the 'Derivative' of the Sigmoid function to see how much we need to change each weight.
   - We send this error back through the hidden layer to adjust the input-to-hidden weights as well.
5. Goal: Over 10,000 iterations, the "Loss" (error) decreases until the model accurately predicts 0 for same inputs and 1 for different inputs.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. ARTIFICIAL NEURAL NETWORK (ANN):
   - A model inspired by the biological brain that consists of layers: Input, Hidden, and Output.
   - The "Hidden Layer" allows the network to learn non-linear relationships.

2. FORWARD PROPAGATION:
   - The process of passing inputs through the network to generate a prediction.
   - Input -> (Weights + Bias) -> Activation -> Output.

3. BACKWARD PROPAGATION:
   - The core algorithm for training neural networks.
   - It calculates the gradient of the error with respect to each weight using the Chain Rule.
   - It effectively "attributes blame" for the error to each weight in the network.


1. WHAT IS BACKPROPAGATION? (As per Lab Manual)
   - It is the most common method for training "Multilayer Perceptrons" (MLP).
   - It is a supervised learning algorithm that works by minimizing the error 
     between the predicted and actual output.
   - It "back-propagates" the error from the output layer through the hidden 
     layers to adjust the weights.

2. WHY IS IT NEEDED? (The XOR Problem):
   - Simple Perceptrons can only solve "Linearly Separable" problems.
   - Real-world problems like XOR are non-linear. Backpropagation with hidden 
     layers allows the network to learn these complex, non-linear boundaries.

3. THE THREE STAGES OF THE BPN ALGORITHM:

   A. FEED-FORWARD STAGE:
      - Input patterns are applied to the input layer.
      - Data flows forward through hidden layers to the output layer.
      - Output is calculated using the Sigmoid activation function.

   B. ERROR BACKPROPAGATION STAGE:
      - The difference between the Target and Predicted output is calculated.
      - This error is multiplied by the derivative of the activation function 
        to find the "Gradient."
      - The error is then sent backward through the weights to the hidden layer.

   C. WEIGHT & BIAS UPDATE STAGE:
      - Weights are modified to reduce the error for the next iteration.
      - Formula: ΔW = Learning_Rate * Error_Gradient * Input.

4. MATHEMATICAL CONCEPTS:
   - Sigmoid Function: f(x) = 1 / (1 + e^-x). It squashes values between 0 and 1.
   - Sigmoid Derivative: f'(x) = x * (1 - x). This is crucial for calculating 
     how much a weight contributed to the total error.
   - Gradient Descent: The process of taking small steps (Learning Rate) down 
     the "Error Hill" to find the minimum point.

5. ADVANTAGES:
   - Capable of learning any complex mapping from input to output.
   - Highly efficient for large datasets and deep architectures.

---------------------------------------------------------------------------------------
                             EXPANDED VIVA Q&A
---------------------------------------------------------------------------------------

Q1: What is a "Multilayer Perceptron" (MLP)?
A1: It is a feed-forward neural network with at least one "Hidden Layer" between 
    the input and output. It uses non-linear activation functions like Sigmoid.

Q2: Why do we use the "Derivative" of the activation function in backprop?
A2: The derivative tells us the "slope" of the error. It helps us understand if 
    we need to increase or decrease the weights to make the error smaller.

Q3: What is the "Vanishing Gradient" problem?
A3: In very deep networks, as the error is propagated backward through many 
    Sigmoid layers, it becomes smaller and smaller (almost zero). This causes 
    the early layers to stop learning.

Q4: Why do we initialize weights with random values instead of zeros?
A4: If weights are zero, all hidden neurons will perform the exact same 
    calculation and receive the exact same updates. This "Symmetry" prevents 
    the network from learning different features.

Q5: What is the "Learning Rate" (η)?
A5: It is a hyperparameter that determines the size of the weight update. A 
    high rate might overshoot the minimum error, while a low rate makes 
    training very slow.

Q6: How does the network handle the XOR problem?
A6: The hidden layer acts as a "Feature Extractor." It transforms the XOR 
    inputs into a new space where they become linearly separable for the 
    output neuron.

Q7: What is an "Epoch" and a "Batch"?
A7: An Epoch is one full pass through the entire training set. A Batch is a 
    small subset of the data used for a single weight update.

Q8: What is "Overfitting" in Backpropagation?
A8: It happens when the network learns the training data "too well" (including 
    noise) and fails to perform accurately on new, unseen data.

---------------------------------------------------------------------------------------
"""
