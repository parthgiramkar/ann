#5: Write a python Program for Bidirectional Associative Memory with two pairs of vectors.

import numpy as np # Import NumPy for matrix math (Outer products and Dot products)

# --- Step 1: Define two pairs of bipolar vectors ---
# Bipolar means we use 1 and -1 instead of 1 and 0.
# Vector Pair 1 (x1 and y1)
x1 = np.array([1, 1, 1, -1])
y1 = np.array([1, -1])

# Vector Pair 2 (x2 and y2)
x2 = np.array([-1, -1, 1, 1])
y2 = np.array([-1, 1])

# --- Step 2: Compute the Weight Matrix (W) ---
# The weight matrix stores the association between X and Y.
# Formula: W = sum of (y_i * x_i_transpose)
# np.outer calculates the outer product of two vectors
W = np.outer(y1, x1) + np.outer(y2, x2)

# --- Step 3: Define the BAM Recall Function ---
def bam(x):
    # Step A: Multiply the Weight matrix by the input vector (Dot product)
    y_raw = np.dot(W, x)
    
    # Step B: Apply the thresholding (Activation)
    # If the result is >= 0, return 1; if it's < 0, return -1
    y_recalled = np.where(y_raw >= 0, 1, -1)
    return y_recalled

# --- Step 4: Testing the BAM Network ---

# Test 1: Testing with a slightly noisy version of x1
# We change one value in x1 to see if the network can still recall y1
x_test = np.array([1, -1, -1, -1]) 
y_recalled = bam(x_test)

print("--- BAM Neural Network Results ---")
print(f"Weight Matrix (W):\n{W}")
print(f"\nInput Vector (x_test): {x_test}")
print(f"Recalled Output (y):   {y_recalled}")

# Verifying if it matched y1 or y2
if np.array_equal(y_recalled, y1):
    print("Result: Successfully recalled Pair 1 (y1)")
elif np.array_equal(y_recalled, y2):
    print("Result: Successfully recalled Pair 2 (y2)")
else:
    print("Result: Recalled an unknown pattern.")





















"""
--- Code Explanation ---
This script implements a Bidirectional Associative Memory (BAM) network.

1. Bipolar Vectors: We use values of 1 and -1. This helps the network distinguish patterns more clearly than 0 and 1.
2. Weight Matrix (W): This is the 'memory' of the network. We create it by adding the 'outer products' of our two pairs. 
   - Effectively, we are 'overlapping' the associations into one matrix.
3. Recall Function: When we give a new 'x', we multiply it by the weights. 
   - The result is then 'thresholded' back into 1s and -1s.
4. Error Correction: As shown in the test, even if the input is slightly different (noisy), the network can often still find the correct matching pair.

---------------------------------------------------------------------------------------
                          THEORY NOTES FOR VIVA (EXTENDED)
---------------------------------------------------------------------------------------

1. WHAT IS BIDIRECTIONAL ASSOCIATIVE MEMORY (BAM)? (As per Lab Manual)
   - Proposed by Bart Kosko in 1988.
   - It is a supervised learning model that acts as a Hetero-associative memory.
   - Unlike basic networks, it returns an output pattern (Y) for an input (X) 
     where X and Y can be of different sizes.
   - It is a type of Recurrent Neural Network (RNN) because it uses feedback loops.

2. SIMILARITY TO THE HUMAN BRAIN:
   - Human memory is naturally associative. We use mental associations to recover 
     lost information (e.g., seeing a face recalls a name).
   - BAM mimics this by using a weight matrix to store "links" between vectors.

3. WHY IS BAM REQUIRED?
   - To store and retrieve hetero-associative pattern pairs.
   - To retrieve a correct pattern even when the input is noisy or incomplete.
   - To provide stability in data association.

4. BAM ARCHITECTURE:
   - It consists of two layers of neurons: Input layer (Set A) and Output layer (Set B).
   - Connections: Every neuron in the X-layer is connected to every neuron in 
     the Y-layer.
   - Bidirectionality: If X is input, it recalls Y. If Y is input, it recalls X 
     using the transpose weight matrix (W^T).

5. THE TRAINING ALGORITHM (Step-by-Step):
   - Step 0: Initialize weights to store pattern pairs using the Correlation Rule.
   - Step 1: Initialize all activations to zero.
   - Step 2: Present the current input pattern 'x' to the X-layer.
   - Step 3: Check for convergence (stop if activations no longer change).
   - Step 4: Update the Y-layer: Calculate net input y_in = Σ (xi * wij).
   - Step 5: Apply activation function (Signum) to get the binary/bipolar output.
   - Step 6: Feedback the signal to the X-layer to update 'x' based on the new 'y'.

6. KEY MATHEMATICAL CONCEPTS:
   - Bipolar Data: Uses (+1, -1). This creates stronger mathematical separation 
     than binary (0, 1), making the memory more robust.
   - Weight Matrix (W): Built using the sum of outer products of stored pairs.
   - Stability: BAM is guaranteed to be stable if the weights are symmetric.

---------------------------------------------------------------------------------------
                             EXPANDED VIVA Q&A
---------------------------------------------------------------------------------------

Q1: What does "Hetero-associative" mean?
A1: It means the network associates an input pattern with a different output 
    pattern (X → Y). This is different from "Auto-associative" where a pattern 
    is associated with itself (X → X).

Q2: Why is it called "Bidirectional"?
A2: Because information can flow in two directions: from X to Y (Forward) and 
    from Y to X (Backward). The same set of weights is used for both.

Q3: What is the significance of the "Equilibrium" state in BAM?
A3: Equilibrium is reached when the vectors 'x' and 'y' stop changing after 
    several loops. This means the network has settled on the most stable memory.

Q4: What happens if the input is "Noisy"?
A4: BAM uses its stored weight correlations to pull the noisy pattern toward 
    the nearest correct memory, effectively "cleaning" the data.

Q5: How do you calculate the Weight Matrix (W)?
A5: By taking the outer product of each associated pair (Y * X_transpose) and 
    summing them all together into a single matrix.

Q6: Can BAM store more patterns than its number of neurons?
A6: No. BAM has a limited storage capacity. If you store too many patterns, 
    they will interfere with each other, leading to "Spurious States" (false memories).

Q7: What is a "Node" in this network?
A7: A node is a single processing unit (neuron) that holds one bit of the 
    pattern (+1 or -1) and performs the threshold calculation.

Q8: Mention two applications of BAM.
A8: 1. Image-to-Text matching (e.g., matching a picture of a car to the word "Car").
    2. Character recognition and pattern data association.

---------------------------------------------------------------------------------------
"""

