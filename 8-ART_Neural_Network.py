# 8: Write a python program to illustrate ART (Adaptive Resonance Theory) neural network.

import numpy as np # Import NumPy for handling pattern arrays and bitwise logic

# --- Define the ART1 Class (Simplified) ---
class ART1:
    def __init__(self, n, m, rho):
        # n: number of input features
        # m: maximum number of clusters (categories) we can form
        # rho: Vigilance parameter (ranges 0 to 1). Controls how strict clustering is.
        self.n = n
        self.m = m
        self.rho = rho
        
        # Initialize Top-Down and Bottom-Up weights
        # Bottom-up (W): From Input to Category
        self.W = np.ones((n, m)) / (1 + n)
        # Top-down (T): From Category back to Input (Template)
        self.T = np.ones((m, n))

    # Activation function for the recognition layer
    def compute_activation(self, x):
        # Calculate the 'match' score for each existing cluster
        return np.dot(x, self.W)

    def train(self, x):
        # Step 1: Compute activations for all clusters
        activations = self.compute_activation(x)
        
        # Step 2: Try to find a match among existing clusters
        # We sort by activation strength to check the 'best' match first
        sorted_indices = np.argsort(activations)[::-1]
        
        for j in sorted_indices:
            # Step 3: Comparison (Vigilance Test)
            # Find the intersection of input and category template
            intersection = np.bitwise_and(x.astype(int), self.T[j].astype(int))
            match_ratio = np.sum(intersection) / np.sum(x)
            
            # If the match is good enough (>= Vigilance rho)
            if match_ratio >= self.rho:
                # Update weights (Learning)
                self.T[j] = intersection
                self.W[:, j] = (intersection.astype(float)) / (0.5 + np.sum(intersection))
                return j # Return the cluster index
        
        return -1 # No match found (in a full model, this would create a new cluster)

# --- Define Test Patterns (Binary) ---
# Each array represents a feature set (e.g., presence of certain traits)
patterns = np.array([
    [1, 1, 0, 0], # Pattern A
    [1, 0, 0, 0], # Pattern B (Similar to A)
    [0, 0, 1, 1], # Pattern C (Different)
    [0, 0, 0, 1]  # Pattern D (Similar to C)
])

# --- Execution ---

# Initialize ART1 with 4 inputs, 2 possible clusters, and 0.5 vigilance
# High rho (e.g. 0.9) makes it very strict; Low rho (e.g. 0.1) makes it very loose.
art = ART1(n=4, m=2, rho=0.5)

print("--- ART Neural Network Clustering Results ---")
for i, x in enumerate(patterns):
    cluster = art.train(x)
    if cluster != -1:
        print(f"Pattern {i} {x} -> Assigned to Cluster {cluster}")
    else:
        print(f"Pattern {i} {x} -> Failed Vigilance Test (No Match)")




















"""
--- Code Explanation ---
This script implements a simplified version of the ART1 (Adaptive Resonance Theory) model.

1. Components:
   - F1 Layer (Input): Receives the binary pattern.
   - F2 Layer (Recognition): Stores categories or 'clusters'.
   - Vigilance (rho): A threshold that decides if an input "belongs" to a cluster or if it's too different.
2. Logic:
   - When a pattern comes in, the network calculates which cluster it matches best.
   - It then performs a 'Vigilance Test'. If the similarity ratio is >= rho, the cluster "claims" the input and updates its template.
   - If no cluster is a good match, the input is rejected (or would normally form a new category).

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS ART (ADAPTIVE RESONANCE THEORY)?
   - Developed by Stephen Grossberg and Gail Carpenter in 1987.
   - It is an "Unsupervised Learning" model. This means it groups data into categories on its own, without humans giving it the "correct" labels.
   - Its main goal is to solve the "Stability-Plasticity Dilemma."

2. THE STABILITY-PLASTICITY DILEMMA:
   - Plasticity: The ability of a brain (or a neural network) to learn NEW things quickly.
   - Stability: The ability to keep OLD memories safe without forgetting them.
   - ART does both: It can learn brand new patterns (plastic) while keeping its old memories perfectly safe (stable).

3. DIFFERENT TYPES OF ART MODELS:
   - ART1: The simplest version. It only works with BINARY data (0s and 1s).
   - ART2: Made to work with CONTINUOUS data (like real numbers and decimals).
   - ART3: Includes chemical-like processes for more advanced learning.
   - ARTMAP: A supervised version that links two different ART models together.

4. THE ARCHITECTURE OF ART:
   - F1 Layer (Comparison Layer): This is where the input data enters and is compared against old memories.
   - F2 Layer (Recognition Layer): This is where the actual categories or clusters are stored.
   - Feedback Loop: The two layers talk back and forth to find the best match.

5. THE 5 PHASES OF LEARNING (Step-by-Step):
   - 1. Initialization: Start with empty clusters.
   - 2. Recognition: F1 sends the input up to F2 to see if it looks familiar (Bottom-up).
   - 3. Comparison: F2 sends its best guess down to F1 (Top-down).
   - 4. Vigilance Test: The network checks if the input is similar enough to the memory. If yes, it's a "Resonance." If no, it's a "Reset" (try another cluster).
   - 5. Learning: Only if they resonate, the weights are updated to remember the new pattern.

6. THE VIGILANCE PARAMETER (ρ):
   - This is the "strictness" setting (a number between 0 and 1).
   - Low Vigilance: The network is "loose." It groups slightly different patterns into large, general clusters.
   - High Vigilance: The network is "strict." It demands almost exact matches, creating many small, specific clusters.

--- Detailed Viva Q&A ---

Q1: What is the main problem that ART tries to solve?
A1: The "Stability-Plasticity Dilemma." It solves the problem of a network forgetting old information when it tries to learn new information.

Q2: What is the difference between ART1 and ART2?
A2: ART1 can only process binary data (0s and 1s). ART2 was built to process continuous, analog data like real numbers.

Q3: What are the two main layers in the ART architecture?
A3: The F1 layer (Comparison Layer) handles the incoming data. The F2 layer (Recognition Layer) holds the actual categories or memory clusters.

Q4: What happens during "Resonance"?
A4: Resonance happens when the incoming data matches a stored memory closely enough to pass the vigilance test. Only when resonance happens does the network actually learn and update its weights.

Q5: What is the role of the "Vigilance Parameter"?
A5: It decides how similar an input must be to a stored memory to belong to that group. High vigilance makes strict, fine-grained categories; low vigilance makes broad, general categories.

Q6: Is ART a supervised or unsupervised learning model?
A6: Basic ART models (like ART1 and ART2) are unsupervised. They group data on their own. However, ARTMAP is a supervised version.

Q7: What happens if an input pattern fails the Vigilance Test for all existing clusters?
A7: The network will just create a completely new, empty cluster to store this unique pattern. This keeps the old memories safe from being corrupted.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: Why did you use `np.bitwise_and` in the code?
   A1: ART1 works on binary bits. The bitwise 'AND' operation is the fastest way to find common features (1s) between the input and the stored memory to check for a match.

   Q2: What does the variable `rho` represent?
   A2: It is the Vigilance parameter. In our code, it's set to 0.5, meaning the input must share at least 50% of its features with a memory to be matched.

   Q3: What is the difference between Top-Down and Bottom-Up weights?
   A3: Bottom-up weights help the network 'Search' for the best category. Top-down weights act as the 'Template' or memory of what that category actually looks like.

   Q4: How would you make the clustering more "strict"?
   A4: By increasing the value of `rho` closer to 1.0. This would force the network to only match patterns that are almost identical.

------------------------- End of Viva Notes -------------------------
"""
