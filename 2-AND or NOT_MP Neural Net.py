# 2: Generate AND NOT function using McCulloch-Pitts neural net.

import numpy as np # Import NumPy for handling arrays and matrix math

# --- Function Definition ---

# This function mimics a single neuron (M-P Model)
def linear_threshold_gate(dot, T):
    # If the combined input (dot) is >= the threshold (T), the neuron fires (1)
    if dot >= T:
        return 1
    else:
        # Otherwise, it stays inactive (0)
        return 0

# --- Preparing the Data ---

# Matrix of inputs (Truth Table for 2 inputs: A and B)
input_table = np.array([
    [0, 0], # A=0, B=0
    [0, 1], # A=0, B=1
    [1, 0], # A=1, B=0
    [1, 1]  # A=1, B=1
])

# For ANDNOT logic:
# We want the output to be 1 ONLY when A is 1 AND B is 0.
# So, A has a positive weight (Exhibitory) and B has a negative weight (Inhibitory).
weights = np.array([1, -1])

# Step 1: Calculate the weighted sum for all inputs using Matrix Multiplication (@)
dot_products = input_table @ weights

# Step 2: Set the threshold (T)
# In ANDNOT, we only fire if the sum is exactly 1 (which only happens for [1, 0])
T = 1

# --- Execution and Output ---

print(f"Input Table:\n{input_table}")
print(f"\nWeights: {weights} | Threshold: {T}\n")

# Loop through each row of the input table to calculate and print the result
for i in range(0, 4):
    activation = linear_threshold_gate(dot_products[i], T)
    print(f"Input: {input_table[i]} | Net Input Sum: {dot_products[i]} | Activation: {activation}")
























"""
--- Code Explanation ---
This script implements the McCulloch-Pitts (M-P) neuron model to solve the ANDNOT logic gate.

1. Inputs: We provide a table of four possible pairs of 0s and 1s.
2. Weights: We assign a weight of '1' to the first input and '-1' to the second. 
   - The -1 acts as an 'Inhibitory' signal, meaning if the second input is '1', 
   it blocks the neuron from firing.
3. Dot Product: We multiply the inputs by their weights and add them up.
4. Threshold (T=1): The neuron only 'fires' (outputs 1) if the final sum is at least 1.

The result is the ANDNOT function: Output is 1 only if the first input is 1 AND the second is 0.


---------------------------------------------------------------------------------------
                          THEORY NOTES FOR VIVA (EXTENDED)
---------------------------------------------------------------------------------------

1. WHAT IS THE MCCULLOCH-PITTS (M-P) NEURON?
   - Introduced by Warren McCulloch and Walter Pitts in 1943.
   - It is the earliest mathematical model of a biological neuron.
   - It is also known as a "Linear Threshold Gate."
   - It classifies inputs into two classes (Binary: 0 or 1).

2. CORE CHARACTERISTICS (As per Lab Manual):
   - Simplistic: It only generates binary outputs (fired or not fired).
   - Fixed Values: Weights and thresholds are set manually, not learned.
   - Binary Logic: Can implement basic logic functions like AND, OR, and NOT.

3. MATHEMATICAL FORMULA:
   - Summation: g(x) = ∑ (xi * wi)
   - Activation: y = f(g(x)) = 1 if g(x) ≥ θ, else 0
   - Here, θ (Theta) represents the threshold value.

4. EXCITATORY VS INHIBITORY WEIGHTS:
   - Excitatory (wi > 0): Helps the neuron reach the threshold to fire. 
     In our code, w1 = 1 is excitatory.
   - Inhibitory (wi < 0): Prevents the neuron from firing. If an inhibitory 
     input is 1, it significantly reduces the weighted sum.
     In our code, w2 = -1 is inhibitory.

5. LOGIC FOR ANDNOT (Input1 AND NOT Input2):
   - The neuron should fire ONLY when Input 1 is TRUE (1) and Input 2 is FALSE (0).
   - Case (1,0): Sum = (1 * 1) + (0 * -1) = 1. Fires because 1 ≥ 1.
   - Case (1,1): Sum = (1 * 1) + (1 * -1) = 0. Silent because 0 < 1. 
     (Input 2 inhibited the neuron).

6. LIMITATIONS OF THE M-P MODEL:
   - No Learning: It cannot update weights automatically based on data.
   - Fixed Threshold: The threshold must be tuned manually for every logic gate.
   - Linear Separation: It can only solve problems that are linearly separable.

---------------------------------------------------------------------------------------
                             EXPANDED VIVA Q&A
---------------------------------------------------------------------------------------

Q1: Who introduced the first model of an artificial neuron?
A1: Warren McCulloch and Walter Pitts in 1943.

Q2: What is an "Inhibitory" input in the M-P model?
A2: It is an input with a negative weight. When this input is 1, it acts like a 
    "stop" signal, making it very difficult for the neuron to reach its 
    threshold and fire.

Q3: Why is M-P called a "Linear Threshold Gate"?
A3: Because it calculates a linear sum of inputs and compares it against a 
    specific threshold value to make a binary decision.

Q4: Can the M-P model solve the XOR problem?
A4: No. XOR is not linearly separable. The M-P model is too simplistic and 
    lacks hidden layers or learning rules required for non-linear logic.

Q5: In your ANDNOT code, what is the role of the weight -1?
A5: It implements the "NOT" part of ANDNOT. It ensures that if the second input 
    is 1, the total sum drops below the threshold, preventing an output of 1.

Q6: What is the significance of the Threshold (θ)?
A6: It acts as the "Decision Boundary." It determines the minimum combined 
    strength the inputs must have before the neuron communicates a signal.

Q7: What is the main difference between M-P and a Perceptron?
A7: The M-P model uses manually fixed weights (no learning). A Perceptron can 
    learn and update its weights automatically using the Perceptron Learning Rule.

Q8: Mention some applications of the M-P model.
A8: Designing simple logical operations and understanding the foundations of 
    how modern neural networks mimic biological brains.

---------------------------------------------------------------------------------------
"""

