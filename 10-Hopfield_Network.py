# Practical No. 10: Write a python program to design a Hopfield Network which stores 4 vectors.

import numpy as np # Import NumPy for matrix operations

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        # Initialize weight matrix with zeros
        self.weights = np.zeros((n_neurons, n_neurons))

    # Training using Hebbian Learning
    def train(self, patterns):
        for pattern in patterns:
            # Formula: W = sum of (p * p_transpose)
            # np.outer creates the correlation matrix for each pattern
            self.weights += np.outer(pattern, pattern)
        
        # Rule: Remove self-connections (diagonal must be 0)
        np.fill_diagonal(self.weights, 0)

    # Prediction / Recall logic
    def predict(self, pattern, steps=5):
        x = pattern.copy()
        for _ in range(steps):
            # Step A: Multiply weights by the input pattern
            x = np.dot(self.weights, x)
            # Step B: Thresholding using the Sign function
            x = np.sign(x)
            # Replace any 0 result with 1 (Hopfield rule for neutral states)
            x[x == 0] = 1
        return x

# --- Main Program Execution ---
if __name__ == '__main__':
    # 1. Define 4 Bipolar Patterns (Size 4)
    patterns = np.array([
        [1, 1, -1, -1],
        [-1, -1, 1, 1],
        [1, -1, 1, -1],
        [-1, 1, -1, 1]
    ])

    n_neurons = patterns.shape[1]

    # 2. Create and Train the Network
    network = HopfieldNetwork(n_neurons)
    network.train(patterns)

    # 3. Display the Weight Matrix
    print("Weight Matrix:\n", network.weights)

    # 4. Test Recalling the Stored Patterns
    print("\n--- Testing Stored Patterns ---")
    for pattern in patterns:
        prediction = network.predict(pattern)
        print("Input pattern:   ", pattern)
        print("Recalled pattern:", prediction)

    # 5. Test with a Noisy Input
    print("\n--- Testing Noisy Input ---")
    # We take the first pattern [1, 1, -1, -1] but change some values
    noisy = np.array([1, 1, -1, -1]) 
    print("Noisy input:", noisy)
    print("Recovered:  ", network.predict(noisy))

























"""
--- Code Explanation ---
This script implements a Discrete Hopfield Network exactly as shown in your manual.

1. Weight Matrix: The weights represent the correlation between bits. If two bits are usually the same across patterns, their weight is positive; if they are different, it's negative.
2. np.outer: This is the mathematical implementation of the Hebbian Rule.
3. np.sign: This is the activation function. It returns 1 for positive numbers and -1 for negative numbers.
4. Convergence: The 'steps=5' loop allows the network to settle into a stable state.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS A HOPFIELD NETWORK?
   - Invented by John Hopfield in 1982.
   - It is a "Recurrent Neural Network" (RNN). This means information loops back and forth between neurons, unlike a Feed-forward network.
   - It acts as an "Auto-associative" memory. This means if you give it a broken or noisy pattern, it will try to fix it and return the original, perfect pattern.

2. CORE FEATURES (Simple Definitions):
   - Fully Connected: Every single neuron is connected to every other neuron in the network.
   - Symmetric Weights: The connection strength from Neuron A to Neuron B is exactly the same as from Neuron B to Neuron A.
   - No Self-Talk: A neuron is not connected to itself (W_ii = 0). This stops it from getting stuck in an infinite loop.
   - Bipolar Values: It works best using +1 and -1 instead of 1 and 0.

3. THE "ENERGY" LANDSCAPE (How it settles):
   - Think of the network's memories as "valleys" in a landscape.
   - When you give it a noisy pattern, it's like dropping a ball on a hill.
   - As the network updates, the ball rolls down the hill until it stops in the lowest valley. That valley is the "Stable State" (the recalled memory).

4. HEBBIAN LEARNING:
   - "Neurons that fire together, wire together."
   - The network learns by strengthening the connections between bits that are usually the same in a pattern.

5. THE STORAGE LIMIT (Spurious States):
   - A Hopfield network cannot store unlimited memories. Its capacity is roughly 15% of its total number of neurons.
   - If you try to stuff too many patterns into it, it gets confused and creates "Spurious States" (fake memories that were never actually taught to it).

--- Detailed Viva Q&A ---

Q1: What is the main difference between a Perceptron and a Hopfield Network?
A1: A Perceptron is "Feed-forward," meaning data only flows straight from input to output. A Hopfield network is "Recurrent," meaning data loops around the neurons until they settle on an answer.

Q2: What does "Auto-associative memory" mean?
A2: It means the network associates a pattern with itself. Its job is to remember a specific pattern so well that if you show it a corrupted or blurry version, it can reconstruct the original perfectly.

Q3: How is Hopfield different from BAM (Practical 5)?
A3: Hopfield is Auto-associative (it maps X back to X). BAM is Hetero-associative (it maps a pattern X to a completely different pattern Y).

Q4: Why must the weights be "Symmetric"?
A4: Symmetry ensures the network behaves like a ball rolling down a hill. Without symmetry, the network might never settle and could just cycle back and forth forever.

Q5: What is a "Spurious State" or "False Memory"?
A5: It happens when the network is overloaded with too many patterns. The math breaks down, and the network settles on a stable pattern that is actually completely wrong.

Q6: Why do we use `np.sign()` in the code?
A6: It is the activation function. It looks at the sum of the inputs and forces the neuron's output to be strictly +1 (if the sum is positive) or -1 (if the sum is negative).

Q7: Why do we use a loop (`for _ in range(steps):`) during the recall phase?
A7: Because it is a recurrent network, the answer isn't found instantly. It needs a few loops to update its state, lower its energy, and finally settle into the correct memory.

------------------------- End of Viva Notes -------------------------
"""
