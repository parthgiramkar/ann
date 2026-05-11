# Practical No. 12: Implement Convolutional Neural Network (CNN) using TensorFlow.

import tensorflow as tf
from tensorflow.keras.datasets import mnist # Dataset of 60,000 handwritten digits
from tensorflow.keras.models import Sequential # To stack layers linearly
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # CNN specific layers
from tensorflow.keras.utils import to_categorical # For one-hot encoding of labels

# --- Step 1: Load and Preprocess the Data ---

# Option A: Using the built-in Keras dataset (Downloads from internet)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Option B: Using a local CSV file (Uncomment these lines if dataset is provided locally)
# import pandas as pd
# from sklearn.model_selection import train_test_split
# df = pd.read_csv("mnist_dataset.csv")
# # Assuming the first column is the label (0-9) and the other 784 columns are the pixels
# y_local = df.iloc[:, 0].values
# X_local = df.iloc[:, 1:].values 
# # Reshape the flat 784-pixel rows back into 28x28 square images
# X_local = X_local.reshape(-1, 28, 28) 
# X_train, X_test, y_train, y_test = train_test_split(X_local, y_local, test_size=0.20, random_state=42)

# Preprocessing:
# 1. Reshape images to (28, 28, 1) -> 28x28 pixels, 1 channel (Grayscale)
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0 # Normalize pixel values to 0-1
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# 2. One-hot encode the labels (e.g., Digit '3' becomes [0,0,0,1,0,0,0,0,0,0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# --- Step 2: Build the CNN Architecture ---

model = Sequential([
    # Layer 1: Convolution (Extracts features like edges/curves)
    # 32 filters of size 3x3
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # Layer 2: Max Pooling (Reduces image size, keeps most important pixels)
    MaxPooling2D((2, 2)),
    
    # Layer 3: Another Convolution (Detects more complex patterns)
    Conv2D(64, (3, 3), activation='relu'),
    
    # Layer 4: Another Pooling
    MaxPooling2D((2, 2)),
    
    # Layer 5: Final Convolution
    Conv2D(64, (3, 3), activation='relu'),
    
    # Layer 6: Flatten (Turns 2D image map into a 1D list of numbers)
    Flatten(),
    
    # Layer 7: Fully Connected Dense layer
    Dense(64, activation='relu'),
    
    # Layer 8: Output Layer (10 neurons for digits 0-9, using Softmax for probability)
    Dense(10, activation='softmax')
])

# --- Step 3: Compile and Train ---

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', # Use this for multi-class classification
    metrics=['accuracy']
)

print("--- Starting CNN Training (MNIST) ---")
# Using only 2 epochs for speed in a practical environment
model.fit(X_train, y_train, batch_size=64, epochs=2, verbose=1)

# --- Step 4: Evaluation ---

print("\n--- Model Evaluation ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")






















"""
--- Code Explanation ---
This script builds a CNN to recognize handwritten digits.

1. Conv2D: Think of this as a 'Scanning Window' that slides over the image to find features like edges and corners.
2. MaxPooling: This shrinks the image size by picking the highest value in a small window, making the model faster and more robust.
3. Flatten: Since the output layer needs a simple list of numbers, we 'flatten' the 2D pixel maps into a 1D array.
4. categorical_crossentropy: This is the loss function used when we have more than 2 classes (here we have 10: digits 0 to 9).
5. Softmax: Used in the final layer to give a percentage 'confidence' for each digit.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS A CNN (CONVOLUTIONAL NEURAL NETWORK)?
   - It is a special type of Deep Learning model built specifically for images and grid-like data.
   - It mimics how human eyes work: it looks for small features first (like a straight line or a curve), and then pieces them together to recognize the whole object (like the number '8').

2. LAYER 1: CONVOLUTION (The "Feature Scanner")
   - Imagine holding a tiny magnifying glass (called a "Filter" or "Kernel") and sliding it across a picture.
   - The filter looks for specific patterns, like edges or corners. When it finds one, it highlights it.

3. LAYER 2: POOLING (The "Shrinker")
   - Max Pooling takes a small block of pixels (like 2x2) and only keeps the brightest (maximum) pixel, throwing the rest away.
   - Why? It shrinks the image size, making the computer's job much faster. It also helps the network recognize a number even if it's drawn slightly off-center.

4. LAYER 3: FLATTENING (The "Bridge")
   - CNNs process images as 2D grids (squares). But the final decision-making neurons can only read a straight, 1D list of numbers.
   - Flattening simply takes the square grid and unrolls it into a single, long line of pixels.

5. RELU VS SOFTMAX (Activation Functions):
   - ReLU: Used in the middle layers. It turns all negative numbers to zero. This speeds up training and helps the network focus only on positive, useful signals.
   - Softmax: Used only at the very end. It looks at the final scores for digits 0-9 and turns them into clean percentages (e.g., 90% sure it's a '7', 10% sure it's a '1').

--- Detailed Viva Q&A ---

Q1: Why is CNN better than a standard Feed-forward Network for images?
A1: A standard network breaks the image into a flat list immediately, which destroys the "shape" and spatial relationships between pixels. A CNN keeps the image as a 2D square, so it understands shapes much better.

Q2: What is a "Kernel" or "Filter" in simple terms?
A2: It is a small mathematical window (like a 3x3 grid) that slides over the image. Its job is to detect a specific feature, like a vertical line or a curved edge.

Q3: Why do we divide the pixels by 255 (Normalization)?
A3: Pixels have values between 0 (black) and 255 (white). Neural networks hate big numbers. By dividing by 255, we shrink all values to be between 0 and 1, which helps the network learn much faster.

Q4: What does "to_categorical" do?
A4: It converts a single number label (like '3') into a binary array (like [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]). This makes it easier for the network to calculate the probabilities for all 10 possible digits.

Q5: What happens to the accuracy if you increase the number of layers?
A5: Usually, more layers mean the network gets "smarter" and accuracy goes up. But if you add too many, the network might just memorize the training images (Overfitting) and fail on new images.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: In `Conv2D(32, (3, 3))`, what do the numbers 32 and (3,3) mean?
   A1: `32` means the network will use 32 different filters (so it can look for 32 different patterns). `(3,3)` means the sliding magnifying glass is 3 pixels wide and 3 pixels tall.

   Q2: Explain `MaxPooling2D((2, 2))` in plain English.
   A2: It looks at a 2x2 square of pixels, picks the brightest one, and discards the other three. This cuts the image height and width perfectly in half.

   Q3: What does the "Flatten" layer actually do?
   A3: It takes the 3D output of the Convolution layers and crushes it into a 1-Dimensional straight line, so the final `Dense` layer can read it.

   Q4: Why do we use `categorical_crossentropy` for the loss function?
   A4: Because this is a "Multi-class" problem (we have 10 digits to choose from). We use Binary Crossentropy when there are only 2 choices, but Categorical when there are 3 or more.

------------------------- End of Viva Notes -------------------------
"""
