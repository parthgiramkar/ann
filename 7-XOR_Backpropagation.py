# 7: Write a python program to show Back Propagation Network for XOR function with Binary Input and Output.

import numpy as np # Import NumPy for handling neural network math

# --- Function Definitions ---

# Sigmoid Activation: Squashes the weighted sum into a 0 to 1 range
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid Derivative: Used to calculate how much to change the weights
def sigmoid_derivative(x):
    return x * (1 - x)

# --- XOR Data (Binary Input and Output) ---

# Inputs: Standard XOR Truth Table (4 cases)
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# Target Outputs: 1 if inputs are different, 0 if they are the same
y = np.array([[0], [1], [1], [0]])

# --- Network Configuration ---

np.random.seed(1) # Seed ensures the random numbers are the same every time you run it

# 1. Initialize Weights with random values
# W1: 2 inputs to 2 hidden neurons | W2: 2 hidden neurons to 1 output
W1 = np.random.uniform(size=(2, 2))
W2 = np.random.uniform(size=(2, 1))

# 2. Initialize Biases
b1 = np.random.uniform(size=(1, 2))
b2 = np.random.uniform(size=(1, 1))

# 3. Learning Parameters
lr = 0.5 # Learning Rate: Controls the speed of learning
epochs = 10000 # Number of training iterations

# --- Training Loop (Backpropagation) ---

print("--- Training XOR Backpropagation Network ---")

for i in range(epochs):
    # PHASE 1: Forward Propagation (The "Guess")
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predicted_output = sigmoid(output_layer_input)

    # PHASE 2: Backward Propagation (The "Learning")
    # 1. Calculate the error at the output
    error = y - predicted_output
    
    # 2. Calculate the "Delta" (gradient) for output layer
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # 3. Backpropagate the error to the hidden layer
    error_hidden_layer = d_predicted_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # 4. Update Weights and Biases (Optimization)
    W2 += hidden_layer_output.T.dot(d_predicted_output) * lr
    W1 += X.T.dot(d_hidden_layer) * lr
    b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    # Print error progress every 2000 epochs
    if i % 2000 == 0:
        print(f"Epoch {i} | Average Error: {np.mean(np.abs(error)):.6f}")

# --- Testing the Network ---
print("\n--- Final XOR Results ---")
for i in range(len(X)):
    print(f"Input: {X[i]} | Predicted: {predicted_output[i][0]:.4f} | Target: {y[i][0]}")







class Linear :


    def __init__( self , input , output ) :


        self.w = np.random.randn( input , output )
        self.b = np.random.randn( 1 , output  )

        self.dx = None
        self.dw = None
        self.db = None


    def forward( self , x ) :


        self.x = x

        z = self.x @ self.w + self.b                  # linear_layers             

        return z
    

    def backward( self , gradient_from_upstream ) :

# gradients wrt to inp , w, b

        self.dx = gradient_from_upstream @ self.w.T

        self.dw =  self.x.T @  gradient_from_upstream 

        self.db = np.sum(gradient_from_upstream , axis=0 , keepdims=True)        # as bias are added acorss diff neurons , so axis=0


        return self.dx
    
    

# %%
class Relu : 
        
        def __init__(self)      :

                self.x = None

        def forward( self , x)   :

                self.x = x

                return np.maximum(self.x , 0 )


        def backward( self , gradient_from_upstream )   :

# loss w.r.t to linearlayer_which is due to Z 

                return gradient_from_upstream * np.where( self.x > 0 , 1 , 0 ) 
                

# %%
class Sigmoid  :

    def __init__(self):
        
        self.sigm = None
        self.x = None

    def forward( self , x ) :
        
        self.x = x
        self.sigm =  1 /  ( 1 + np.exp(-self.x) )

        return self.sigm
    
    def backward( self , gradient_from_upstream ) :

# loss w.r.t to linearlayer_2 , which is due to Z2 

        return gradient_from_upstream * self.sigm * ( 1 - self.sigm )

# %%
class MSE :


    def __init__(self) :
     
        self.ypred = None
        self.yactual = None


    def forward( self , yactual , ypred ) :

        self.ypred = ypred
        self.yactual = yactual

        loss = np.mean( 0.5 * (yactual - ypred )**2 )                # .mean() means we_have taken each_valuefrom_data

        return loss


    def backward( self ) :

# loss w.r.t to ypred_which is due to activation function -i.e relu

        n = np.prod(self.yactual.shape)            # return the dimn,which_gotmultplied
        return  ( self.ypred - self.yactual ) / n



# %%
class training :
    

    def __init__(self , input_neurons , hidden_neurons , output_neurons ) :
        
# passing on to_the layers ,here are 2layers

        self.layer1 = Linear( input_neurons , hidden_neurons ) 
        self.relu_activation = Relu()
        self.layer_2 = Linear( hidden_neurons , output_neurons )
        self.sigmoid_activation = Sigmoid()
        self.loss_funct = MSE()
        self.loss_history = []                # plotting purpose


    def train( self , x , y ) :
         
        
        epochs = 10000
        learning_rate = 0.1

        for i in range( epochs ) :

        # weights and bias for first ip layer 
        
# Forward Pass
            
            z_1 = self.layer1.forward(x)
            relu_1 = self.relu_activation.forward(z_1)


        # weights and bias for hidden layer 
            z_2 = self.layer_2.forward(relu_1)
            sigm = self.sigmoid_activation.forward(z_2)

            loss = self.loss_funct.forward( y ,  sigm )
            self.loss_history.append(loss)
            
            if i%500 == 0 :
                print(f"Loss at {i}th epoch - {loss}")

# Backward Pass

            grad_relu_2 = self.loss_funct.backward()               # loss w.r.t to activation - Relu (layer2)

            grad_Z_2 = self.sigmoid_activation.backward(grad_relu_2)                     

            grad_relu_1 = self.layer_2.backward(grad_Z_2)            # loss w.r.t to layer1st relu activation func

            grad_Z_1 = self.relu_activation.backward(grad_relu_1)

            grad_X = self.layer1.backward(grad_Z_1)          # loss wrt to input , calc w and bias also
            
# optimisation using gradient descent algo

            self.layer1.w = self.layer1.w - learning_rate * self.layer1.dw
            self.layer1.b = self.layer1.b - learning_rate * self.layer1.db

            self.layer_2.w = self.layer_2.w - learning_rate * self.layer_2.dw
            self.layer_2.b = self.layer_2.b - learning_rate * self.layer_2.db


        print("Loss at 999th epoch - ",loss)
        print("\n Final Input Weight " , self.layer1.w)
        print("\n Final Input bias ", self.layer1.b)

        print("\n Final Hidden Weight " , self.layer_2.w)
        print("\n Final Hidden bias ", self.layer_2.b)



    def predict( self , x ) :

# Forward pass for prediction purpose

        z1 = self.layer1.forward(x)
        relu1 = self.relu_activation.forward(z1)

        z2 = self.layer_2.forward(relu1)
        sigm = self.sigmoid_activation.forward(z2)
  
        return sigm                         # Y_prediction                 



# %%
# Loading the input and target values


if __name__ ==  "__main__" :

    np.random.seed(5)
# The xor logic gate logic

    x = np.array([
        [0,0] , [0,1] , [1,0] , [1,1]              # size - ( 4,2)
    ])

    y = np.array(
        [0,1,1,0]  
    )                     #  the xor results size - ( 4 ) , its a list , convert it into 2dform

    y = y.reshape( -1 , 1)            # now shape - ( 4 , 1 ) , column vector


    input_neurons = 2
    hidden_neurons = 4
    output_neurons = 1

    trn = training( input_neurons , hidden_neurons , output_neurons )
    trn.train( x , y)
    

# %%
trn.predict(x)

# %%
ans = trn.predict(x)
np.round(ans)

# %%


# %%
predicted_values = np.round(ans)


actual_values = ( y == predicted_values)        # == , means if the values are same , then o/p - 1 , else 0
accuracy = np.mean( actual_values ) * 100
print(f"Final Model accuracy - {accuracy}%")

# %%


# %%
import matplotlib.pyplot as plt

plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve (0.01)")
plt.plot(trn.loss_history)






















"""
--- Code Explanation ---
This script implements a 3-layer Neural Network (Input, Hidden, Output) to solve XOR.

1. Sigmoid: We use this because XOR outputs are binary (0/1). It's a 'differentiable' function.
2. W1, W2: These are matrices representing the strength of connections between layers.
3. Forward Pass: We multiply inputs by weights, add bias, and apply sigmoid.
4. Error Calculation: We find the difference between our guess and the real target.
5. Backward Pass: We use the 'Chain Rule' (via Sigmoid Derivative) to update the weights.
6. Convergence: After 10,000 passes, the weights settle into values that perfectly mimic the XOR logic.

-----
---------------------------------------------------------------------------------------
                          THEORY NOTES FOR VIVA (EXTENDED)
---------------------------------------------------------------------------------------

1. OBJECTIVE:
   - To implement a Backpropagation Network (BPN) specifically for the XOR function.
   - The inputs and outputs are Binary (0 and 1).

2. THE XOR PROBLEM (Non-Linearity):
   - XOR is a "Non-Linearly Separable" function. 
   - Unlike AND or OR gates, you cannot separate XOR's 0s and 1s with a single 
     straight line.
   - This was a famous limitation of early neural models (like M-P or Perceptrons).
   - Solution: Using a Hidden Layer between the input and output.


3. THE TWO PHASES:
   - Forward Pass: Data moves from left to right to produce an output.
   - Backward Pass: Error moves from right to left to adjust weights.

4. WEIGHTS & BIASES:
   - Weights: Control the "Slope" or "Angle" of the decision boundary.
   - Biases: Control the "Offset" or "Position" of the decision boundary.


3. HOW BACKPROPAGATION SOLVES XOR:
   - The hidden layer acts as a "Feature Transformer."
   - It takes the 2D input (0,0, 0,1, etc.) and projects it into a higher-dimensional 
     space where the output neuron can finally draw a line to separate them.

4. BPN ARCHITECTURE FOR XOR:
   - Input Layer: 2 neurons (for two binary inputs).
   - Hidden Layer: Usually 2 to 4 neurons (to learn the non-linear features).
   - Output Layer: 1 neuron (for the binary result).

5. KEY STEPS IN THE ALGORITHM:
   - Step 1: Forward Pass - Calculate the activation of hidden neurons, then 
     the output neuron using the Sigmoid function.
   - Step 2: Error Calculation - Find the difference between the XOR target 
     and the actual output.
   - Step 3: Backward Pass - Calculate the gradient of the error and propagate 
     it back to update the weights of both the hidden and output layers.

6. ACTIVATION FUNCTION (SIGMOID):
   - Formula: f(x) = 1 / (1 + e^-x)
   - It is crucial because it is "Differentiable," allowing us to calculate 
     the gradients needed for weight updates.

---------------------------------------------------------------------------------------
                             EXPANDED VIVA Q&A
---------------------------------------------------------------------------------------

Q1: Why can't a simple Perceptron solve the XOR problem?
A1: Because XOR is not linearly separable. A Perceptron can only solve problems 
    where a single line can separate the classes. XOR requires a non-linear 
    decision boundary.

Q2: What is the role of the "Hidden Layer" in this practical?
A2: The hidden layer captures the "interaction" between the inputs. It transforms 
    the data so that the XOR pattern becomes clear to the output layer.

Q3: What is "Gradient Descent" in BPN?
A3: It is the process of adjusting weights in the opposite direction of the 
    error gradient to "slide down" to the minimum possible error.

Q4: What happens if you use a "Linear" activation function instead of Sigmoid?
A4: Multiple linear layers mathematically collapse into a single linear layer. 
    The network would remain linear and would still fail to solve XOR.

Q5: Why is the learning rate usually kept small (e.g., 0.1 to 0.5)?
A5: To ensure the network converges smoothly. If it's too high, the weights 
    might change so drastically that the model never finds the correct XOR pattern.

Q6: How do you define the "Error" in this code?
A6: Error = (Target XOR Output) - (Predicted XOR Output). This value tells 
    the network how far off its guess was.

Q7: What is the "Symmetry Breaking" problem?
A7: If we start with all weights at zero, every neuron learns the same thing. 
    Starting with small random weights (Symmetry Breaking) allows each hidden 
    neuron to learn a different part of the XOR logic.

Q8: Mention one real-world application of Backpropagation.
A8: Speech recognition, image classification, and any task involving complex 
    pattern matching that isn't easily solved by simple rules.

---------------------------------------------------------------------------------------
"""

