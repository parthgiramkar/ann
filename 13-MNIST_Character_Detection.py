# Practical No. 13: Implementation of MNIST Handwritten Character Detection using TensorFlow/Keras.

import tensorflow as tf # The main Deep Learning framework
import numpy as np # Import NumPy for numerical operations
from tensorflow.keras.datasets import mnist # The 0-9 handwritten digits dataset
from tensorflow.keras.models import Sequential # For stacking layers
from tensorflow.keras.layers import Dense, Flatten # Standard ANN layers
from tensorflow.keras.optimizers import Adam # The optimization algorithm

# --- Step 1: Load and Preprocess the Dataset ---

# Option A: Using the built-in Keras dataset (Downloads from internet)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Option B: Using a local CSV file (Uncomment these lines if dataset is provided locally)
# import pandas as pd
# from sklearn.model_selection import train_test_split
# df = pd.read_csv("mnist_dataset.csv")
# # Assuming the first column is the label (0-9) and the other 784 columns are the pixels
# y_local = df.iloc[:, 0].values
# X_local = df.iloc[:, 1:].values
# # Reshape the flat 784-pixel rows back into 28x28 square images (our Flatten layer expects a 2D image)
# X_local = X_local.reshape(-1, 28, 28)
# X_train, X_test, y_train, y_test = train_test_split(X_local, y_local, test_size=0.20, random_state=42)

# Normalization: Scaling pixel values from [0-255] down to [0-1]
# This makes the math easier for the network to calculate.
X_train = X_train / 255.0
X_test = X_test / 255.0

# --- Step 2: Define the Model Architecture ---

# This is an ANN (Multi-Layer Perceptron) approach
model = Sequential([
    # 1. Flatten Layer: Converts the 28x28 image grid into a single list of 784 numbers.
    Flatten(input_shape=(28, 28)),

    # 2. Hidden Layer: 128 neurons to learn the complex patterns of the digits.
    Dense(128, activation='relu'),

    # 3. Output Layer: 10 neurons (one for each digit 0-9).
    # Softmax gives the probability for each class.
    Dense(10, activation='softmax')
])

# --- Step 3: Compile the Model ---

model.compile(
    optimizer=Adam(learning_rate=0.001),
    # We use 'sparse' because our labels are simple integers (0, 1, 2...)
    # instead of one-hot vectors.
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Step 4: Train the Model ---

print("--- Starting Character Detection Training ---")
# Training for 5 epochs (full passes through the data)
model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=1)

# --- Step 5: Evaluate Accuracy ---

print("\n--- Final Model Evaluation ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# --- Step 6: Single Prediction Demo ---
prediction = model.predict(X_test[:1])
print(f"Prediction for the first test image: {np.argmax(prediction)}")
print(f"Actual label: {y_test[0]}")


























"""
--- Code Explanation ---
This script detects handwritten digits using an Artificial Neural Network (ANN).

1. Flatten: Images are 2D squares (28x28). The ANN needs them in a 1D line (784 pixels). Flatten does this conversion.
2. ReLU: This activation function helps the neurons 'fire' only for important pixel patterns.
3. Softmax: Turns the output of the 10 neurons into a list of 10 percentages. The highest percentage is our prediction.
4. Adam Optimizer: A very popular "driver" that updates weights efficiently.
5. Evaluation: We test the network on images it has NEVER seen before to see its real-world performance.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS AN ANN (ARTIFICIAL NEURAL NETWORK)?
   - It is the most basic type of neural network, also known as a Multi-Layer Perceptron.
   - It consists of an Input layer, Hidden layers, and an Output layer.
   - All layers are "Dense" (every neuron is connected to every other neuron).

2. THE "FLATTEN" LAYER (The Crusher):
   - Images are naturally 2D squares (28x28 pixels).
   - An ANN does not understand squares. It only understands straight lines.
   - The Flatten layer crushes the 2D image into a single 1D string of 784 pixels so the network can read it.

3. ANN vs CNN (Why do this practical?):
   - CNNs (Practical 12) are smarter because they keep the image as a square and understand shapes.
   - ANNs (This practical) destroy the shape by flattening it immediately. 
   - We learn this to understand the "baseline." It proves that even a "dumb" network can guess standard, perfectly centered digits with high accuracy.

4. SOFTMAX (The Probability Converter):
   - It is used only in the final output layer.
   - We have 10 output neurons (for digits 0-9). Softmax forces their final scores to sum up to exactly 100%. 
   - The neuron with the highest percentage is the model's final guess.

5. LOSS FUNCTION (Sparse Categorical Crossentropy):
   - "Categorical" means we are choosing between many categories (0 to 9).
   - "Sparse" is a handy trick. It means we can just feed the network normal numbers (like `7`) instead of massive binary matrices (like `[0,0,0,0,0,0,0,1,0,0]`).

--- Detailed Viva Q&A ---

Q1: What is the MNIST dataset?
A1: It is a famous database containing 70,000 small (28x28 pixel) grayscale images of handwritten digits (0 through 9). It is the standard "Hello World" of machine learning.

Q2: Why must we use the "Flatten" layer first?
A2: Because standard Dense neural networks can only accept 1-Dimensional data. If we don't flatten the 2D image, the network will crash because it doesn't know how to read a grid.

Q3: What is the difference between "Softmax" and "Sigmoid"?
A3: Sigmoid is for "Yes or No" (Binary). Softmax is for "Which one of these?" (Multiple choice). Softmax guarantees that all the choices add up to 100%.

Q4: What does the "Adam" optimizer do?
A4: It is the "driver" that fixes the weights when the network makes a mistake. It is highly efficient because it automatically speeds up or slows down its learning rate depending on the situation.

Q5: Why is the accuracy usually very high (over 95%) even for this basic model?
A5: Because the MNIST digits are perfectly centered and clean. If the digits were messy, off-center, or rotated, this basic ANN would fail, and we would be forced to use a CNN.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: What does `Dense(128)` do?
   A1: It creates a hidden layer with 128 artificial neurons. These neurons act as the "brain," trying to find patterns in the 784 input pixels.

   Q2: Why do we divide the training data by 255 (`X_train / 255.0`)?
   A2: This is called Normalization. Pixel colors range from 0 to 255. Neural networks hate large numbers. Dividing by 255 shrinks all values to be between 0 and 1, making the math much faster and more stable.

   Q3: If the output layer has 10 neurons, how do we know the final answer?
   A3: We look at which of the 10 neurons has the highest probability. If neuron number 4 has a 98% probability, the network's final guess is the digit '4'.

   Q4: What is `model.evaluate()`?
   A4: It runs the trained network against the "Test Dataset" (images it has never seen before) to prove that it actually learned the shapes and didn't just memorize the training data.

------------------------- End of Viva Notes -------------------------
"""
