# 4. With a suitable example demonstrate the perceptron learning law with its decision regions 
# using python. Give the output in graphical form.

import numpy as np
import matplotlib.pyplot as plt

# --- Data Preparation Section ---

# CHOICE A: Using scikit-learn (ACTIVE)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, [0, 2]] # Taking Sepal Length and Petal Length
y = np.where(iris.target == 0, 0, 1) # Binary: Setosa vs Others

# CHOICE B: If a CSV dataset is provided, UNCOMMENT the lines below:
# import pandas as pd
# data = pd.read_csv('iris.csv')
# X = data.iloc[:, [0, 2]].values # Select columns for features
# y = np.where(data.iloc[:, -1] == 'Setosa', 0, 1) # Convert labels to 0 and 1

# --- Perceptron Parameters ---

w = np.zeros(2) # Initial weights for our 2 features
b = 0 # Initial bias
lr = 0.1 # Learning Rate (eta)
epochs = 30 # Number of times to repeat the training

# --- Training the Model ---

# Perceptron Prediction function
def predict(x):
    # If (X * W) + Bias >= 0, return 1, else return 0
    return np.where(np.dot(x, w) + b >= 0, 1, 0)

# Training Loop: Adjusting weights based on error
for _ in range(epochs):
    for i in range(len(X)):
        y_pred = predict(X[i]) # Predict for current data point
        error = y[i] - y_pred  # Calculate error (Target - Prediction)
        
        # Update weights: wj(t+1) = wj(t) + η(d-y)x
        w[:] = w + lr * error * X[i]
        b = b + lr * error

# --- Plotting Decision Regions ---

# 1. Create a grid to cover the entire plot area
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# 2. Predict the class for every single point on the grid
Z = predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) # Reshape back to the grid size

# 3. Draw decision regions and scatter data points
plt.contourf(xx, yy, Z, alpha=0.4)

# 4. Plot the actual data points on top
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')

# Labels and Title
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Perceptron Decision Regions (Iris Setosa vs Others)")
plt.show()
























"""
--- Code Explanation ---
This practical visualizes how a Perceptron "sees" and separates data.

1. Load Data: We use the Iris dataset, specifically looking at Sepal and Petal lengths.
2. Binary Target: We simplify the 3-class problem into "Is it a Setosa or not?".
3. Training: We use a loop where the model makes a guess, finds the error, and adjusts its 'w' and 'b'.
4. Meshgrid: We create a fine grid of points across the graph.
5. Contourf: We color the background based on the model's prediction for every point on the grid.
   - This creates the 'Decision Region' (the area where the model would predict '0' vs '1').

---------------------------------------------------------------------------------------
                          THEORY NOTES FOR VIVA (EXTENDED)
---------------------------------------------------------------------------------------

1. WHAT IS A NEURAL NETWORK? (As per Lab Manual)
   - It is a branch of Artificial Intelligence loosely modeled after the human brain.
   - It acts like a "Black Box" that learns to predict output patterns when 
     presented with input patterns.
   - It has the ability to learn from experience and adapt to changing environments.

2. PERCEPTRON LEARNING LAW:
   - Originally developed by Frank Rosenblatt in the late 1950s.
   - It is a supervised learning algorithm for binary classifiers.
   - The connection weights are modified by an amount proportional to the 
     difference between the actual output and the desired output.

3. THE MATHEMATICAL FORMULA:
   - Summation: Sum = f(∑ wi * xi + b)
   - Weight Update: wj(t+1) = wj(t) + η(d - y)x
   - (Where d is desired output, y is actual output, and η is the gain/step size).

4. PERCEPTRON LEARNING ALGORITHM STEPS:
   - Step 1: Initialize weights and threshold to small random numbers (or zero).
   - Step 2: Present an input vector 'x' to the neuron.
   - Step 3: Calculate the output.
   - Step 4: If the output is incorrect, update weights and bias using the formula.
   - Step 5: Repeat until the error is below a threshold or iterations are complete.

5. DECISION REGIONS:
   - These are the areas in the feature space where a classifier predicts a 
     specific class.
   - The linear decision boundary separates the two classes (e.g., +1 and -1).

6. APPLICATIONS:
   - Supervised learning of binary classification tasks.
   - Pattern recognition and data categorization.

---------------------------------------------------------------------------------------
                             EXPANDED VIVA Q&A
---------------------------------------------------------------------------------------

Q1: What do you mean by a "Neural Network" in terms of learning?
A1: It is a system that can learn from experience to improve its performance. 
    It doesn't follow strict pre-defined rules but instead adapts its internal 
    weights to recognize patterns in data.

Q2: Explain the "Weight Update" logic in simple terms.
A2: If the model makes a correct guess, we do nothing. If it's wrong, we add 
    or subtract a small portion of the input from the weights to "steer" the 
    model toward the correct answer.

Q3: What is the purpose of the decision regions in your plot?
A3: They visually show the "territory" of each class. The background colors 
    show where the model would classify any new, unknown point.

Q4: Why was the Perceptron a breakthrough in the 1950s?
A4: It was the first algorithm that could automatically "learn" the importance 
    (weights) of different inputs rather than requiring a human to set them.

Q5: What is the significance of the Learning Rate (η) in this practical?
A5: It is the "step size" of learning. The manual suggests a range of 0.0 < η < 1.04. 
    It determines how much a single mistake changes the model's mind.

Q6: When does the Perceptron algorithm stop?
A6: It stops when the "Iteration Error" becomes zero (all points correctly 
    classified) or after a predetermined number of epochs.

Q7: What is the main component of the M-P model that Perceptron improved?
A7: The M-P model had fixed weights. Perceptron introduced the ability to 
    LEARN those weights automatically from data.

Q8: Mention two applications of Perceptrons.
A8: Binary classification (Yes/No tasks) and being the fundamental building 
    block for Deep Neural Networks.

---------------------------------------------------------------------------------------
"""
