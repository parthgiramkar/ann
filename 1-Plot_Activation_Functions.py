#1. Write a Python program to plot a few activation functions that are being used in neural networks.

import numpy as np # Import NumPy for math and handling arrays (lists of numbers)
import matplotlib.pyplot as plt # Import Matplotlib for drawing the graphs

# --- Function Definitions (The Math Behind Activation) ---

# Sigmoid Function: Converts any number into a value between 0 and 1
def sigmoid(x):
    # Math: 1 divided by (1 + e to the power of negative x)
    return 1 / (1 + np.exp(-x))

# ReLU Function: Keeps positive numbers as they are, turns negative numbers to 0
def relu(x):
    # np.maximum compares 0 and x, and picks the bigger one
    return np.maximum(0, x)

# Tanh Function: Converts any number into a value between -1 and 1
def tanh(x):
    # Uses the built-in hyperbolic tangent function from NumPy
    return np.tanh(x)

# Softmax Function: Converts numbers into probabilities (summing up to 1)
def softmax(x):
    # Step 1: Calculate e^x for every number (exp)
    # Step 2: Divide each result by the total sum of all results
    return np.exp(x) / np.sum(np.exp(x))

# --- Creating the Data and the Layout ---

# Create 100 points between -10 and 10 to serve as our X-axis
x = np.linspace(-10, 10, 100)

# Create a 2x2 grid of graphs (4 total) with a size of 8x8 inches
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# --- Plotting each function ---

# Top-Left: Plot the Sigmoid curve
axs[0, 0].plot(x, sigmoid(x))
axs[0, 0].set_title('Sigmoid') # Give this specific graph a title

# Top-Right: Plot the ReLU (Rectified Linear Unit) line
axs[0, 1].plot(x, relu(x))
axs[0, 1].set_title('ReLU')

# Bottom-Left: Plot the Tanh (Hyperbolic Tangent) curve
axs[1, 0].plot(x, tanh(x))
axs[1, 0].set_title('Tanh')

# Bottom-Right: Plot the Softmax probabilities
axs[1, 1].plot(x, softmax(x))
axs[1, 1].set_title('Softmax')

# --- Final Polishing of the Graphs ---

# Set a main title for the entire window
fig.suptitle('Common Activation Functions Visual Comparison')

# Loop through all 4 graphs to add labels for X and Y axis
for ax in axs.flat:
    ax.set(xlabel='Input (x)', ylabel='Output (y)')

# Adjust the space between graphs so they don't overlap
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Finally, pop up the window to show the plots
plt.show()


























"""
--- Code Explanation ---
This script helps visualize how different 'activation functions' work in machine learning.

1. Sigmoid: Turns any number into a value between 0 and 1. It's like a smooth "S" curve.
2. ReLU (Rectified Linear Unit): If the number is negative, it becomes 0. If it's positive, 
   it stays the same.
3. Tanh: Similar to Sigmoid, but it stretches the output to be between -1 and 1.
4. Softmax: Often used at the very end of a model to turn numbers into probabilities 
   (that add up to 1).

The script creates a 2x2 grid of graphs to show these functions visually for numbers ranging from 
-10 to 10.

------------------------- Detailed Theory Notes for Viva -------------------------

1. DEFINITION:
   - An Activation Function is a mathematical formula applied to the output of a neural network layer.
   - It determines whether a specific neuron should be "activated" (turned on) or 
     not based on the input it receives.
   - Essentially, it acts as a gate that decides which information is important enough to be passed 
     to the next layer.

2. WHY DO WE NEED THEM? (THE IMPORTANCE OF NON-LINEARITY):
   - Real-world data is complex and rarely follows a straight line.
   - Without activation functions, a neural network is just a giant linear regression model 
     (a straight line).
   - Non-linear functions allow the model to learn complex shapes, curves, and patterns 
     (like recognizing a face or a car).

3. TYPES OF ACTIVATION FUNCTIONS:
   - Linear: The output is proportional to the input (e.g., y = x). Rarely used in hidden layers.
   - Non-Linear: These allow the model to generalize and learn complex data. 
     Examples: Sigmoid, ReLU, Tanh.

4. COMMON NON-LINEAR FUNCTIONS (The "Big Four"):

   A. SIGMOID (Logistic Function)
      - Definition: Squashes any input value into a range between 0 and 1.
      - Math: f(x) = 1 / (1 + e^-x)
      - Example Use: Used in the output layer of a "Yes/No" (Binary) classification problem 
        (e.g., Is this email spam?).
      - Drawback: It can make learning very slow if the numbers are very high or very low 
        (Vanishing Gradient).

   B. TANH (Hyperbolic Tangent)
      - Definition: Similar to Sigmoid but squashes values between -1 and 1.
      - Example Use: Often used in hidden layers because it is "zero-centered," meaning the average 
        output is close to 0, which helps the model learn faster.

   C. ReLU (Rectified Linear Unit)
      - Definition: If the input is positive, it outputs the same number. If it's negative, 
        it outputs 0.
      - Math: f(x) = max(0, x)
      - Example Use: The "Default" choice for hidden layers in almost all modern deep learning models 
        (CNNs, LLMs).
      - Advantage: It's super fast to calculate and doesn't suffer as much from slow learning as 
        Sigmoid.

   D. SOFTMAX
      - Definition: Takes a group of numbers and turns them into probabilities that add up to 1 
        (or 100%).
      - Example Use: Used in the very last layer for "Multi-class" problems 
        (e.g., Is this a cat, dog, or bird?).

5. KEY VIVA TERMS TO KNOW:
   - Vanishing Gradient: When the "signal" for learning becomes so small that the model stops improving.
   - Zero-Centered: When the function's output range is balanced around zero (like Tanh), 
     which helps optimization.
   - Dead ReLU: A problem where some neurons "die" and always output 0 because they only receive 
     negative inputs.

--- Expanded Viva Q&A ---

Q1: What is the main role of an activation function?
A1: Its primary role is to introduce non-linearity into the network, allowing it to solve 
    complex problems that simple linear models cannot.

Q2: If you are building a model to predict if an image is a "Cat" or "Not a Cat," which 
    function would you use at the end?
A2: Sigmoid, because it gives a probability between 0 and 1, which is perfect for 
    two-choice (binary) problems.

Q3: What if you are choosing between Cat, Dog, and Rabbit?
A3: Softmax, because it will give probabilities for all three classes that sum up to 1.

Q4: Why is ReLU preferred over Sigmoid in hidden layers?
A4: ReLU is much faster to compute and helps avoid the vanishing gradient problem, allowing deep networks to train more effectively.

6. PRACTICAL & CODING VIVA QUESTIONS (About your script):

   Q1: What does `np.linspace(-10, 10, 100)` do in your code?
   A1: It creates an array of 100 numbers equally spaced between -10 and 10. This gives us the "x-axis" data for our plots.

   Q2: Why did you use `plt.subplots(2, 2)`?
   A2: This command creates a layout of 4 graphs (2 rows and 2 columns) so we can compare all four activation functions in a single window.

   Q3: Looking at your plots, which function is only active for positive values?
   A3: ReLU. It stays at 0 for any negative input and only starts increasing when the input is greater than 0.

   Q4: Which libraries are essential for this practical?
   A4: NumPy (for mathematical operations and arrays) and Matplotlib (for creating the visual graphs).

   Q5: In your `softmax` function, why do we use `np.exp(x)`?
   A5: We use the exponential function to ensure all outputs are positive and to make larger values significantly more prominent than smaller ones before calculating probabilities.

------------------------- End of Viva Notes -------------------------
"""
