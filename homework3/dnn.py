# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:49:23 2020

@author: Roman AVJ

Implement a DNN. Code isnspired by the Cpurse of deeplearning.AI of Coursera
given by Andrew NG.
"""
import numpy as np

# =============================================================================
# Private functions
# =============================================================================

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    # activation function
    A = 1/(1+np.exp(-Z))
    
    #save linear part
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    # activation function
    A = np.maximum(0,Z)
    
    # save linear part
    cache = Z 
    
    return A, cache

def g_activation(Z, activation):
    """
    Implement the activation function

    Arguments:
    Z -- Output of the linear layer, of any shape
    activation -- The activation function. Can be ReLU or Sigmoid. 
    
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    if activation == 'sigmoid':
        A, cache = sigmoid(Z)
    elif activation == 'relu':
        A, cache = relu(Z)
    else:
        raise('Activation Function do not match')
    
    return A, cache
    
    

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0,  should set dz to 0 as well. 
    # dA g'(x)
    dZ[Z <= 0] = 0
    
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    # g(x)
    s = 1/(1+np.exp(-Z))
    
    # dA g'(x)
    dZ = dA * s * (1-s)
    
    
    return dZ

def g_backward(dA, cache, activation):
    """
    Implement the backward propagation for one layer for g(.)

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    activation -- The activation function. Can be ReLU or Sigmoid. 

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, cache)
    else:
        raise('Not an activation function implemented')
    
    return dZ
    
    
    
    
# =============================================================================
# DNN functions
# =============================================================================
## Part1: Init params
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(8)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
                
    return parameters

## Part2: Forward Propagation
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    # maes broadcasting with b
    Z = W @ A + b
    
    # save parameters
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    # compute linear part
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    # compute activation 
    A, activation_cache = g_activation(Z, activation=activation)
    
    # save params
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    # init parameters
    caches = []
    A = X
    # number of layers in the neural network
    L = len(parameters) // 2       
    
    # compute forward prop for layers
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation='relu')
        caches.append(cache)

    
    # Implement LINEAR -> SIGMOID. Output layer.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation='sigmoid')
    caches.append(cache)
            
    return AL, caches

## Part3: Cost Function
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    # Compute loss from AL and y.
    cost = - np.mean(Y * np.log(AL) + (1 - Y) * np.log(1- AL))

    # To make sure your cost's shape is what we expect.
    cost = np.squeeze(cost)     

    return cost

## Part4: Back propagation
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    # get values used in forward prop
    A_prev, W, b = cache
    m = A_prev.shape[1]

    # compute gradient
    dW = dZ @ A_prev.T / m
    db = np.mean(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    # get values
    linear_cache, activation_cache = cache
    
    # backward prop for one layer
    dZ = g_backward(dA, activation_cache, activation=activation)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    # init params
    grads = {}
    L = len(caches) # the number of layers
    Y = Y.reshape(AL.shape) # Y assure same shape as AL
    
    ## Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    
    # Lth layer (SIGMOID -> LINEAR) gradients. 
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, 
                                                                                                    current_cache, 
                                                                                                    activation='sigmoid'
                                                                                                   )
    # Loop from l=L-2 to l=0
    # lth layer: (RELU -> LINEAR) gradients.
    for l in reversed(range(L-1)):
        # get activation values of the l layer 
        current_cache = caches[l]
        
        # compute back prop for layer l
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)],
                                                                   current_cache,
                                                                   activation='relu')
        
        # save gradients of the l layer
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

## Part5: Update parameters with the dradient
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

## Part6: Assemble 
def dnn(X, Y, layers_dims, learning_rate = 0.0075, epochs = 3000):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (p, n), where p number of vars and n number of observations
    Y -- y label with 1 and 0, of shape (1, n)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    epochs -- number of iterations of the optimization loop
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    # init vars
    costs = {}
    
    # Part1: Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, epochs):

        # Part2: Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)
        
        # Part3: Compute cost
        cost = compute_cost(AL, Y)
    
        # Part4: Backward propagation
        grads = L_model_backward(AL, Y, caches)
 
        # Part5: Update parameters
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
        
        # Part6: Append costs per 100
        if i % 500 == 0:
            costs[str(i)] = cost
        
    
    return parameters, costs

## Part7: Predict
def predict(X, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    p = np.where(probas > 0.5, 1, 0)
        
    return p