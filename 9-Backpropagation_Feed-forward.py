# Practical No. 9: Write a python program for creating a Back Propagation Feed-forward neural network.

import numpy as np # Import NumPy for handling layers as matrices

# --- Define the Generalized Neural Network Class ---
class FeedForwardNN:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        # We define the size of each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # Initialize Weights randomly
        # W1: Connects Input to Hidden | W2: Connects Hidden to Output
        self.W1 = np.random.randn(self.input_nodes, self.hidden_nodes)
        self.W2 = np.random.randn(self.hidden_nodes, self.output_nodes)
        
        # Initialize Biases with zeros
        self.b1 = np.zeros((1, self.hidden_nodes))
        self.b2 = np.zeros((1, self.output_nodes))

    # Activation Function: Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of Sigmoid for Backpropagation
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # PHASE 1: Forward Pass (Input -> Output)
    def forward(self, X):
        # Calculate Hidden Layer values
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Calculate Output Layer values
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.sigmoid(self.z2)
        return self.y_hat

    # PHASE 2: Backward Pass (Output Error -> Update Weights)
    def backward(self, X, y, y_hat, lr):
        # 1. Output Error
        self.error = y - y_hat
        
        # 2. Gradient for Output Layer
        delta2 = self.error * self.sigmoid_derivative(y_hat)
        
        # 3. Gradient for Hidden Layer (Error back-flow)
        delta1 = delta2.dot(self.W2.T) * self.sigmoid_derivative(self.a1)
        
        # 4. Update Weights and Biases
        self.W2 += self.a1.T.dot(delta2) * lr
        self.W1 += X.T.dot(delta1) * lr
        self.b2 += np.sum(delta2, axis=0, keepdims=True) * lr
        self.b1 += np.sum(delta1, axis=0, keepdims=True) * lr

    # Training Function
    def train(self, X, y, epochs, lr):
        for i in range(epochs):
            y_hat = self.forward(X)
            self.backward(X, y, y_hat, lr)
            
            # Print average absolute error every 1000 steps
            if i % 1000 == 0:
                print(f"Epoch {i} | Loss: {np.mean(np.abs(self.error)):.6f}")

# --- Step 3: Example Usage (Logic Gate pattern) ---

# Inputs: 3 features (Binary)
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
# Output: 1 target
y = np.array([[0], [1], [1], [0]])

# Create Network: 3 inputs -> 4 hidden neurons -> 1 output
nn = FeedForwardNN(input_nodes=3, hidden_nodes=4, output_nodes=1)

# Train the network
print("--- Training Generalized Feed-forward Network ---")
nn.train(X, y, epochs=5000, lr=0.1)

# Test Output
print("\n--- Testing Model Predictions ---")
final_out = nn.forward(X)
for i in range(len(X)):
    print(f"Input {X[i]} -> Output {final_out[i][0]:.4f}")



























"""
--- Code Explanation ---
This script creates a flexible 'Multi-Layer Feed-forward' neural network.

1. Layer Nodes: The network size is set at initialization (e.g., 3 inputs, 4 hidden, 1 output).
2. Weights (W1, W2): These are the matrices that hold the network's knowledge. They are randomly initialized.
3. Forward() method: Implements the 'Feed-forward' part. Inputs are multiplied by weights and passed through the Sigmoid activation function to get a result.
4. Backward() method: Implements 'Backpropagation'. It calculates the error at the end and uses calculus (derivatives) to fix the weights in reverse order.
5. dot() and dot(W2.T): NumPy functions used to perform matrix math, which is how data travels between layers.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS A FEED-FORWARD NEURAL NETWORK (FFNN)?
   - It is the simplest type of artificial neural network.
   - "Feed-forward" means the data only moves in ONE direction: from the input, through the hidden layers, and finally to the output.
   - There are no cycles or feedback loops (unlike BAM or Recurrent networks).

2. THE THREE MAIN LAYERS:
   - Input Layer: The "sensors" that receive the raw data (like pixels of an image or numbers from a dataset).
   - Hidden Layer(s): The "brain" of the network. It extracts patterns and complex features from the raw data.
   - Output Layer: The final prediction or decision made by the network.

3. STAGE 1: THE FORWARD PASS (Making a Guess)
   - Step A: Multiply the inputs by the Weights and add the Bias.
   - Step B: Pass that result through an Activation Function (like Sigmoid).
   - Step C: Send the output to the next layer until you reach the end.

4. STAGE 2: THE BACKWARD PASS (Fixing the Errors)
   - Step A: Calculate the Loss (Error). How far off was the network's guess from the true answer?
   - Step B: Use "Gradient Descent" to figure out which weights caused the error.
   - Step C: Apply the "Chain Rule" from calculus to send the error backward and update the weights so the network does better next time.

5. WHY IS THIS IMPORTANT?
   - Universal Approximation Theorem: A feed-forward network with just one hidden layer can learn to solve almost any mathematical or logical problem, as long as it has enough neurons.

--- Detailed Viva Q&A ---

Q1: What does "Feed-forward" mean in this network?
A1: It means that information only flows forward. The data goes from input to hidden to output, without ever looping back on itself.

Q2: Why do we need the "Sigmoid" activation function?
A2: It is a non-linear curve. Without it, the network could only solve simple straight-line problems. Also, it is "differentiable," which means we can easily calculate its slope to fix our errors during backpropagation.

Q3: What is "Gradient Descent" in simple terms?
A3: Imagine walking down a mountain while blindfolded. Gradient Descent is the algorithm that tells the network which way is "down" (toward zero error) so it can adjust its weights to get there.

Q4: What is the purpose of the "Bias" in the code?
A4: Bias is like an anchor or an offset. It allows the network's activation curve to shift left or right, giving it the flexibility to fit patterns that don't perfectly align with zero.

Q5: What is "MSE" or Mean Squared Error?
A5: It is a formula used to calculate how wrong the network's predictions are. We take the difference between our guess and the real answer, square it (to remove negative signs), and find the average.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: Why did you use `np.random.randn` instead of `np.random.rand`?
   A1: `randn` generates numbers that can be both positive and negative (a Gaussian distribution). This helps the network learn faster and avoids it getting stuck early on.

   Q2: What is the "Learning Rate" (lr = 0.1)?
   A2: It is the "step size" the network takes when fixing its weights. A small step (0.1) means it learns slowly but carefully. A large step means it learns fast but might miss the best solution.

   Q3: What would happen if we didn't have a Hidden Layer?
   A3: The network would just be a basic Perceptron. It would fail at any complex task (like XOR) because it couldn't learn non-linear patterns.

   Q4: How do you know when to stop training?
   A4: We stop training when the Error gets very close to zero, or after a set number of "Epochs" (like 5000 loops) where the weights have stabilized.

------------------------- End of Viva Notes -------------------------
"""
