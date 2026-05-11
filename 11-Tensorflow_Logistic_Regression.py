# Practical No. 11: Train a Neural Network and evaluate Logistic Regression using TensorFlow.

import tensorflow as tf # Import TensorFlow for building the neural network
import numpy as np # Import NumPy for numerical operations
from sklearn.model_selection import train_test_split # To split data into training and testing sets
from sklearn.preprocessing import StandardScaler # To scale/normalize the data
from sklearn.datasets import load_breast_cancer # The dataset we will use for classification

# --- Step 1: Load and Prepare the Dataset ---

# Option A: Using the built-in scikit-learn dataset (Breast Cancer - Binary classification)
data = load_breast_cancer()
X = data.data
y = data.target

# Option B: Using a local CSV file (Uncomment these lines if dataset is provided locally)
# import pandas as pd
# df = pd.read_csv("dataset.csv")
# X = df.iloc[:, :-1].values  # All columns except the last one are features
# y = df.iloc[:, -1].values   # The last column is the target label (0 or 1)

# Split the data: 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# IMPORTANT: Scaling the data. Neural networks work best when inputs are in a similar range.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Step 2: Build the Neural Network Model ---

# Logistic Regression = 1 Layer + 1 Neuron + Sigmoid Activation
model = tf.keras.models.Sequential([
    # Dense(1) means one neuron. Input_shape matches the number of features.
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

# --- Step 3: Compile the Model ---

model.compile(
    optimizer='adam',               # Adam is an efficient optimization algorithm
    loss='binary_crossentropy',     # Use this loss for binary (0 or 1) classification
    metrics=['accuracy']            # We want to track accuracy during training
)

# --- Step 4: Train the Model ---

print("--- Starting Training ---")
# Epochs: The number of times the model sees the entire dataset
model.fit(X_train, y_train, epochs=15)

# --- Step 5: Evaluate the Model ---

print("\n--- Model Evaluation ---")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")






























"""
--- Code Explanation ---
This script builds a simple neural network that performs 'Logistic Regression' using TensorFlow.

1. Dataset: We use the Breast Cancer dataset, which has 30 features and a 0/1 target.
2. StandardScaler: This is a crucial step! It scales the data so that all features have a mean of 0 and variance of 1. Without this, the model might learn very slowly or fail.
3. Sequential Model: This is a linear stack of layers.
4. Dense(1, activation='sigmoid'): This creates a single neuron. The 'sigmoid' turns any output into a probability between 0 and 1.
5. Binary Crossentropy: This is the mathematical formula used to measure how 'wrong' the probability guesses are.
6. Adam: This is the 'driver' that adjusts the weights automatically to reduce the error.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. LOGISTIC REGRESSION IN TENSORFLOW:
   - What is it? Logistic Regression is actually the simplest possible Neural Network.
   - It is a network with ZERO hidden layers and exactly ONE output neuron.
   - It is used for "Supervised Binary Classification" (predicting either 0 or 1, Yes or No).

2. THE SIGMOID FUNCTION (The Magic Math):
   - It is the core of this algorithm.
   - Formula: f(x) = 1 / (1 + e^-x).
   - What does it do? It takes any raw number (even very large or very small) and "squashes" it into a probability score between 0 and 1.

3. DATA SCALING (StandardScaler):
   - Why do we need it? Neural networks get confused if features are on different scales (like Age = 20 vs Salary = 50,000). The larger number will unfairly dominate the training.
   - Scaling forces all features to have a mean of 0 and a variance of 1, so the network treats them equally and learns much faster.

4. LOSS FUNCTION (Binary Crossentropy):
   - This calculates how "wrong" the model is.
   - If the real answer is 1, and the model guesses 0.99, the loss is very low. If the model guesses 0.01, the loss (penalty) is extremely high.

5. OPTIMIZER (Adam):
   - Adam stands for 'Adaptive Moment Estimation'.
   - It is the "driver" that automatically tweaks the weights to lower the loss. It is generally faster and smarter than basic Gradient Descent.

--- Detailed Viva Q&A ---

Q1: Why is it called "Logistic Regression" if it does "Classification"?
A1: Because under the hood, it calculates a "Logistic probability score" (a continuous regression value like 0.85). We then apply a threshold (usually 0.5) to classify it into discrete buckets (e.g., 0.85 becomes 1).

Q2: What is the difference between `model.fit()` and `model.evaluate()`?
A2: `fit()` is for TRAINING. The model looks at the training data, checks its errors, and updates its weights. `evaluate()` is for TESTING. The model takes a final exam on data it has never seen before to prove it actually learned.

Q3: What does the term "Epochs" mean?
A3: An epoch is one complete, full pass through the entire training dataset. If epochs=15, the model reviews the data 15 times to keep improving.

Q4: What happens if we don't use the Sigmoid activation function?
A4: The model would just output raw numbers (like -5 or 42) instead of a probability. It would become a "Linear Regression" model and would fail at classification.

Q5: Why do we split the data into Training and Testing sets?
A5: To prevent "Overfitting" (when the model just memorizes the answers). The test set acts as an unseen exam to verify that the model can generalize to new data.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: What does `Dense(1)` mean in the code?
   A1: `Dense` means a fully connected layer. The `1` means we are only using a single artificial neuron for this entire network.

   Q2: What does `input_shape=(X_train.shape[1],)` do?
   A2: It tells the network exactly how many features (columns) to expect in the data. `shape[1]` automatically counts the columns.

   Q3: What is "Accuracy"?
   A3: It is the percentage of correct predictions. An accuracy of 90% means the model correctly guessed 90 out of 100 cases in the test set.

   Q4: Can this script handle 3 or more classes (like Red, Green, Blue)?
   A4: No. Because we use a single neuron with Sigmoid and Binary Crossentropy, this code is strictly for 2-class (Binary) problems.

------------------------- End of Viva Notes -------------------------
"""
