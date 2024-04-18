#Import the necessary libraries
import numpy as np
import pandas as pd
import torch
import torchvision as tv
import matplotlib.pyplot as plt

#To give necessary certificates (necessary for dataset loading to not fail due to error, for me at least)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#Shortcut for tensors
tensors = tv.transforms.ToTensor()
#Code to pull dataset from torchvision(to calculate mean and std)
CiFarDataX = tv.datasets.CIFAR10(root = 'data', download = True)
FashionDataX = tv.datasets.FashionMNIST(root = 'data', download= True)

#Mean
CIFAR10Mean = np.mean(CiFarDataX.data, axis = (0,1,2))/255
FashionMean = np.mean(np.array(FashionDataX.data), axis = (0,1,2))/255

#Standard Deviation
CIFAR10STD = np.std(CiFarDataX.data, axis = (0,1,2))/255
FashionSTD = np.std(np.array(FashionDataX.data), axis = (0,1,2))/255

#Now we have the necessary values to normalize (checked with https://github.com/kuangliu/pytorch-cifar/issues/19) ******DELETE********

#Cifar10 Data
CIFAR10_TrainData = tv.datasets.CIFAR10(root = 'data', train = True, download = True, transform = tv.transforms.Compose([tensors , tv.transforms.Normalize(CIFAR10Mean,CIFAR10STD)]))
CIFAR10_TestData = tv.datasets.CIFAR10(root = 'data', train = False, download = True, transform = tv.transforms.Compose([tensors , tv.transforms.Normalize(CIFAR10Mean,CIFAR10STD)]))

#Fashion Data
Fashion_TrainData = tv.datasets.FashionMNIST(root = 'data', train = True, download = True, transform = tv.transforms.Compose([tensors , tv.transforms.Normalize(FashionMean,FashionSTD)]))
Fashion_TestData = tv.datasets.FashionMNIST(root = 'data', train = False, download = True, transform = tv.transforms.Compose([tensors , tv.transforms.Normalize(FashionMean,FashionSTD)]))

Fashion_TestData_Loader = torch.utils.data.DataLoader(Fashion_TrainData, batch_size = 10000)
FashionX, FashionY = next(iter(Fashion_TestData_Loader))
FashionX = FashionX.numpy().astype(np.float64, casting = 'same_kind')
FashionY = FashionY.numpy()

#Normal Function
def normal(x):
    return x

#Relu Function
def Relu(input):
    return np.maximum(0,input)

#Relu Derivative (*** Should this depend on if w*x if > 0 or just the thing we are taking the derivative off)
def reluD(x):
    return np.where(x <= 0, 0, 1)

#SoftmaxFunction
def softmax(x):
    top = np.exp(x - np.max(x, axis = 1, keepdims = True))
    return top/np.sum(top, axis = 1, keepdims=True)

#SoftmaxGradient
def softmax_gradient(y_true, x, W):
    
    #First we compute the softmax probabilities
    y_pred = softmax(np.dot(x,W))

    #Set the weight gradient to an array with the same shape as the weight array
    gradient = np.zeros(np.shape(W), dtype = np.float64) #Must sepcify type or there will be an error (why I don't know)

    #So we calculate the weight gradient for each neuron and each class ( W is a F * C )
    for neurons in range(len(W)):
        for samples in range(len(x)):
            gradient[neurons] = gradient[neurons] + x[samples][0][neurons] * (y_pred[samples][0] - y_true[samples])
        gradient[neurons] = gradient[neurons] * (1/len(x))

    return gradient

class MLP:

    def __init__(self, weights, activation_functions, gradient_function, gradients, num_hidden_layers, units_in_hidden_layers, num_class, initiaization = 'Uniform', minibatches_size = 100):
        #TO-DO: Make a function to initialize weights, so all we have to pass in weights field is an empty array of the right size to fill (OPTIONAL)
        
        self.weights = weights
        self.activationlist = activation_functions
        self.gradientlist = gradients
        self.gradient = gradient_function
        self.num_hidden = num_hidden_layers
        self.units_in_hidden = np.append([1],units_in_hidden_layers) #For use in the gradient calculations
        self.num_class = num_class
        self.minibatches_size = minibatches_size
        
    def Y_reclassify(self, Y):

        #If Y is not currently one-hot encoded for softmax use, return an array that is
        new_Y = np.zeros((len(Y), self.num_class), dtype = np.float64)
        for i in range(len(Y)):
            new_Y[i][int(Y[i])] = 1
        return new_Y
        
        
    #Function to train model and call optimizer to update weights during training
    def fit(self, X, Y, learning_rate, max_iter, epsilon):
        

        #One-hot y since we will be using softmax at the end
        Y = self.Y_reclassify(Y)

        #For when to stop (epsilon) *******************TODOOOOOOOOOO****************
        norms = np.array([np.inf])
        
        t = 0

        while ((t < max_iter)):
            #To-Do: Add a way to stop due to epsilon and weight gradients

            #Setup for SGD
            n_test = int((len(Y)*(1/self.minibatches_size)))
            inds = np.random.permutation(len(Y))

            #Loop for SGD
            for i in range(n_test):
                Y_Train, X_Train = Y[inds[i*n_test:n_test*(i+1)]], X[inds[i*n_test:n_test*(i+1)]] #Current minibatch to run gradient
                
                current = X_Train
                
                print(np.zeros((len(current),self.units_in_hidden[0]), dtype = np.float64).shape)
                print(np.zeros((len(current),self.units_in_hidden[1]), dtype = np.float64).shape)
                
                #This is where the error is!!!!!!!
                neurons = np.array(([np.zeros((len(current),self.units_in_hidden[l]), dtype = np.float64)] for l in range(self.num_hidden + 1)))

                print(neurons.shape)

                for layer in range(self.num_hidden):
                    #Calculate the neurons of each level
                    current = np.dot(current, self.weights[layer])
                    neurons[layer + 1] = current #Should be of same shape if we did everything right (if you're getting an error make sure all dtype = np.float64)

                g = self.gradient(neurons = neurons, activations = self.gradientlist, unit_layers = self.units_in_hidden, Y = Y_Train, weights = self.weights)

                #Update gradient for each layer
                for layer in range(len(self.weights)):
                    # To-DO: Add regularization (may have been already done by Joyce, you could check with her)
                    self.weights[layer] = (self.weights[layer] - (learning_rate * (g[layer])))

            t += 1
            #********* Just to check on progress, can be easily removed if you don't want to keep it ************
            if (t % 20 == 0):
                print(t)

        return t
    
    #Function to predict class based on current weights
    def predict(self, X):

        #Go through all the functions to get probabilities for each class (always ends with softmax)
        i = 0
        for function in self.activationlist:
            if (i == 0):
                current = function(np.dot(X, self.weights[i]))
            else:
                current = function(np.dot(current, self.weights[i]))
                i += 1

        #Get predictions from softmax based on highest probability
        predictions = np.zeros(len(X))
        for samples in range(len(current)):
            max = 0
            for classes in current[0]:
                if (current[samples][classes] > current[samples][max]):
                    max = classes
            predictions[samples] = max + 1
        
        return predictions





def runExp():
    print('Experiment 1: Different Initializations')
    Exp1()

def Exp1():

    N,D = FashionX[0][0].shape
    Sw = np.zeros((128,10), dtype = np.float64)
    Rw = np.zeros((N,D,128), dtype = np.float64)

    w = [Rw,Sw]
    model = MLP(weights = w, activation_functions = [Relu], gradient_function = backpropagation, gradients= [reluD,normal], num_hidden_layers = 1, units_in_hidden_layers = [128], num_class = 10)
    model.fit(FashionX,FashionY,0.01,2,0.0000001)
    print('Zero Initialization: ', model.weights)

    Sw = np.random.uniform(low = -1, high = 1, size = (128,10), dtype = np.float64)
    Rw = np.random.uniform(low = -1, high = 1, size = (N,D,128), dtype = np.float64)

    w = [Rw,Sw]

    model = MLP(weights = w,activation_functions = [Relu, softmax], gradient_function = backpropagation, num_hidden_layers = 1, units_in_hidden_layers = [128], num_class = 10, minibatches = 100)
    model.fit(FashionX,FashionY,0.01,2,0.0000001)
    print('Uniform Initialization: ', model.weights)

def backpropagation(neurons, activations, unit_layers, Y, X, weights):
    # Neurons is an array of neurons of each layer (index 0 is empty array that is not meant to be used (useless))
    # Activations is a list of activation functions derivatives (Length of hidden layers + 1, ex: 2 hidden layers, relu activations => [reluD, reluD, normal])
    # Unit_Layers is the number of neurons at each hidden layer (length of hidden layers + 1, unit_layers[0] = 1 to tell to switch to tensordot)
    # Y samples (assumed to already be onehot encoded)
    # Weights is an array of weights at each layer

    #Make zero array to store gradient
    g = np.zeros((weights.shape), dtype = np.float64)

    divider = 1/len(neurons[0]) #For efficiency
    length = len(neurons[0]) #For efficiency

    #So we are going to calculate the output derivative and go backwards
    i = 0
    for layers in range(len(neurons)).__reversed__:

        #We always start with the softmax gradient values
        if (i == 0):

            #Make softmax predictions
            y_pred = softmax(np.dot(neurons[layers],weights[layers])) #Should give array of shape (length, 10)

            #Make a saved array of zeros
            saved = np.zeros((length, unit_layers[layers]), dtype = np.float64) #Should give array of shape (length, 10)

            #Start loop
            for samples in range(length):
                #We save first portion of gradient calculation to saved array
                saved[samples] = y_pred[samples][0] - Y[samples] #Should give array of shape (10,)

                #Normal cross entropy loss derivative
                g[layers] = g[layers] + (neurons[layers][samples] * (saved[samples])) #Should give array of shape (128,10) *****There may be an error here

            g[layers] = g[layers] * divider # Needs to be divided by number of samples
            i += 1 #Ensure we dont calculate this again (definitly a better way to do this but meh)

        else:
            checker = unit_layers[layers] #Used to check if we are at last layer

            if (checker == 1):
                #For final new_saved, we need a shape to handle the tensors! 
                new_saved = np.zeros((length, X.shape), dtype = np.float64)
            else:
                #Make saved array of shape (# of samples, # of neurons of said layer)        
                new_saved = np.zeros((length, checker), dtype = np.float64)
            
            for samples in range(length):

                if (checker != 1):
                    # dot product of Saved * activations [previous layer] (weights of previous layer (transposed))
                    new_saved[samples] = np.dot(saved[samples], activations[layers-1](weights[layers+1]).T) #Should give array of shape (# of units in current layer,)
                    # New_Saved * activations[current layer] (neurons of current layer)
                    g[layers] = g[layers] + activations[layers](neurons[layers][samples]) * (new_saved[samples])
                else:
                    new_saved[samples] = np.tensordot(saved[samples], activations[layers-1](weights[layers+1]).T)
                    # New_Saved * X (tensor form) (neurons of current layer)
                    g[layers] = g[layers] + X[samples][0] * (new_saved[samples]) #Should return array of type (28,28,#neurons of first hidden layer) OR (30,30, #neurons of first hidden layer)

            g[layers] = g[layers] * divider #Always end with division by num of smaples

            saved = new_saved.copy #Make copy of new_saved in saved to pass forward (may be a better way of doing this but again, meh)

    return g


#Line so we run experiments
runExp()