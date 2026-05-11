# 6: Implement Artificial Neural Network training process in Python by using Forward Propagation, Back Propagation.

import numpy as np # Import NumPy for matrix operations and random weight generation

# --- Define the Neural Network Class ---
class NeuralNetwork:
    # 1. Initialize the network with Input, Hidden, and Output layers
    def __init__(self, input_size, hidden_size, output_size):
        # Weights (W1: Input to Hidden | W2: Hidden to Output)
        # We use random initialization because starting with zeros prevents learning
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        
        # Biases (b1: Hidden layer | b2: Output layer)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    # 2. Activation Function (Sigmoid): Squashes values between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 3. Sigmoid Derivative: Used to calculate the gradient during Backpropagation
    def sigmoid_derivative(self, x):
        # Formula: sigmoid(x) * (1 - sigmoid(x))
        return x * (1 - x)

    # 4. Forward Propagation: Moving the data from Input to Output
    def forward_propagation(self, X):
        # Layer 1 (Hidden): dot product + bias, then apply activation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2 (Output): dot product + bias, then apply activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.sigmoid(self.z2)
        
        return self.y_hat

    # 5. Backward Propagation: Moving the ERROR from Output back to Input to adjust weights
    def backward_propagation(self, X, y, y_hat, lr):
        # Step A: Calculate Output Layer Error
        self.error = y - y_hat
        # Calculate Delta (Error * Derivative)
        self.delta2 = self.error * self.sigmoid_derivative(y_hat)
        
        # Step B: Calculate Hidden Layer Error (sending delta2 back through W2)
        self.a1_error = self.delta2.dot(self.W2.T)
        self.delta1 = self.a1_error * self.sigmoid_derivative(self.a1)
        
        # Step C: Update Weights and Biases using the gradients
        # New Weight = Old Weight + (Inputs * Delta)
        self.W2 += self.a1.T.dot(self.delta2) * lr
        self.b2 += np.sum(self.delta2, axis=0, keepdims=True) * lr
        self.W1 += X.T.dot(self.delta1) * lr
        self.b1 += np.sum(self.delta1, axis=0) * lr

    # 6. Training Function: Running Forward and Backward passes for many iterations
    def train(self, X, y, epochs, lr):
        for i in range(epochs):
            y_hat = self.forward_propagation(X)
            self.backward_propagation(X, y, y_hat, lr)
            
            # Print the error every 1000 epochs to track progress
            if i % 1000 == 0:
                loss = np.mean(np.square(y - y_hat)) # Mean Squared Error
                print(f"Epoch {i} | Loss: {loss:.6f}")

# --- Step 7: Define XOR Data ---
# XOR cannot be solved by a simple Perceptron (Needs a hidden layer)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# --- Step 8: Execution ---

# Create a network: 2 Inputs -> 4 Hidden Neurons -> 1 Output
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Train the network
print("--- Starting Training (XOR Problem) ---")
nn.train(X, y, epochs=10000, lr=0.1)

# Final Predictions
print("\n--- Final Predictions after 10,000 Epochs ---")
predictions = nn.forward_propagation(X)
for i in range(len(X)):
    print(f"Input: {X[i]} | Predicted: {predictions[i][0]:.4f} | Target: {y[i][0]}")











import numpy as np

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
This script implements a Multi-Layer Perceptron (MLP) to solve the XOR logic gate.

1. Architecture: 2 Inputs -> 1 Hidden Layer (4 neurons) -> 1 Output.
2. Random Weights: We use `np.random.randn` because if all weights are 0, the neurons will all learn the exact same thing (Symmetry problem).
3. Forward Prop: We calculate the weighted sum and apply Sigmoid at each layer to move the signal forward.
4. Backward Prop (The "Learning" Part):
   - We calculate the error at the output.
   - We use the 'Derivative' of the Sigmoid function to see how much we need to change each weight.
   - We send this error back through the hidden layer to adjust the input-to-hidden weights as well.
5. Goal: Over 10,000 iterations, the "Loss" (error) decreases until the model accurately predicts 0 for same inputs and 1 for different inputs.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. ARTIFICIAL NEURAL NETWORK (ANN):
   - A model inspired by the biological brain that consists of layers: Input, Hidden, and Output.
   - The "Hidden Layer" allows the network to learn non-linear relationships.

2. FORWARD PROPAGATION:
   - The process of passing inputs through the network to generate a prediction.
   - Input -> (Weights + Bias) -> Activation -> Output.

3. BACKWARD PROPAGATION:
   - The core algorithm for training neural networks.
   - It calculates the gradient of the error with respect to each weight using the Chain Rule.
   - It effectively "attributes blame" for the error to each weight in the network.


1. WHAT IS BACKPROPAGATION? (As per Lab Manual)
   - It is the most common method for training "Multilayer Perceptrons" (MLP).
   - It is a supervised learning algorithm that works by minimizing the error 
     between the predicted and actual output.
   - It "back-propagates" the error from the output layer through the hidden 
     layers to adjust the weights.

2. WHY IS IT NEEDED? (The XOR Problem):
   - Simple Perceptrons can only solve "Linearly Separable" problems.
   - Real-world problems like XOR are non-linear. Backpropagation with hidden 
     layers allows the network to learn these complex, non-linear boundaries.

3. THE THREE STAGES OF THE BPN ALGORITHM:

   A. FEED-FORWARD STAGE:
      - Input patterns are applied to the input layer.
      - Data flows forward through hidden layers to the output layer.
      - Output is calculated using the Sigmoid activation function.

   B. ERROR BACKPROPAGATION STAGE:
      - The difference between the Target and Predicted output is calculated.
      - This error is multiplied by the derivative of the activation function 
        to find the "Gradient."
      - The error is then sent backward through the weights to the hidden layer.

   C. WEIGHT & BIAS UPDATE STAGE:
      - Weights are modified to reduce the error for the next iteration.
      - Formula: ΔW = Learning_Rate * Error_Gradient * Input.

4. MATHEMATICAL CONCEPTS:
   - Sigmoid Function: f(x) = 1 / (1 + e^-x). It squashes values between 0 and 1.
   - Sigmoid Derivative: f'(x) = x * (1 - x). This is crucial for calculating 
     how much a weight contributed to the total error.
   - Gradient Descent: The process of taking small steps (Learning Rate) down 
     the "Error Hill" to find the minimum point.

5. ADVANTAGES:
   - Capable of learning any complex mapping from input to output.
   - Highly efficient for large datasets and deep architectures.

---------------------------------------------------------------------------------------
                             EXPANDED VIVA Q&A
---------------------------------------------------------------------------------------

Q1: What is a "Multilayer Perceptron" (MLP)?
A1: It is a feed-forward neural network with at least one "Hidden Layer" between 
    the input and output. It uses non-linear activation functions like Sigmoid.

Q2: Why do we use the "Derivative" of the activation function in backprop?
A2: The derivative tells us the "slope" of the error. It helps us understand if 
    we need to increase or decrease the weights to make the error smaller.

Q3: What is the "Vanishing Gradient" problem?
A3: In very deep networks, as the error is propagated backward through many 
    Sigmoid layers, it becomes smaller and smaller (almost zero). This causes 
    the early layers to stop learning.

Q4: Why do we initialize weights with random values instead of zeros?
A4: If weights are zero, all hidden neurons will perform the exact same 
    calculation and receive the exact same updates. This "Symmetry" prevents 
    the network from learning different features.

Q5: What is the "Learning Rate" (η)?
A5: It is a hyperparameter that determines the size of the weight update. A 
    high rate might overshoot the minimum error, while a low rate makes 
    training very slow.

Q6: How does the network handle the XOR problem?
A6: The hidden layer acts as a "Feature Extractor." It transforms the XOR 
    inputs into a new space where they become linearly separable for the 
    output neuron.

Q7: What is an "Epoch" and a "Batch"?
A7: An Epoch is one full pass through the entire training set. A Batch is a 
    small subset of the data used for a single weight update.

Q8: What is "Overfitting" in Backpropagation?
A8: It happens when the network learns the training data "too well" (including 
    noise) and fails to perform accurately on new, unseen data.

---------------------------------------------------------------------------------------
"""
