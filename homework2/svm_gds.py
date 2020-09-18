# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:56:06 2020

@author: Ryo
Stochastic gradient descendent
"""
import numpy as np
import matplotlib.pyplot as plt


def loss_01(theta, theta0, x, y):
    if y * (np.dot(x, theta) + theta0) < 1 :
        return(-1)
    else:
        return(0)

def hinge_loss(theta, theta0, x, y):
    z = y * (np.dot(x, theta) + theta0)
    if z < 1:
        return(1-z)
    else:        
        return(0)
    
def J_loss(theta, theta0, X, Y, lambd):
    # tuple of row vectors of the matrix
    X_tuple = tuple(X)

    # vector of the loss 0/1 for each xi, yi 
    lossh_vec = np.array([hinge_loss(theta, theta0, xi, yi) for xi, yi in zip(X_tuple,Y)])
    
    # J loss
    J = lossh_vec.mean() + lambd / 2 * np.linalg.norm(theta) ** 2
    return(J)


def svm_sgd(X, Y, rho, lambd, tol, max_iter):
    # fixed values
    #number of observations and number of variables
    p = np.shape(X)[1]
    n = np.shape(X)[0]
        
    # initialize
    theta0 = np.array(0).reshape(1)
    theta = np.zeros((1,p))
    k = 0
    
    # save data
    J_vec = np.array(0)
    J_vec = np.append(J_vec, J_loss(theta[k], theta0[k], X, Y, lambd))
    
    while abs(J_vec[k+1] - J_vec[k]) > tol and k < max_iter:
        # stochastic index 
        index_batch = range(1, n)
        size_stochastic = np.random.choice(index_batch)
        j = list(np.random.choice(index_batch, size=size_stochastic))
        
        #subset stochastic subset
        X_stochastic = X[j]
        Y_stochastic = Y[j]
        Xs_tuple = tuple(X_stochastic)
        
        # inner operations
        loss01_y = np.array([yi * loss_01(theta[k], theta0[k], xi, yi) 
                             for xi, yi in zip(Xs_tuple,Y_stochastic)])
        vec_LyX = (X_stochastic.T * loss01_y).T
        
        # update
        thet = theta[k] - rho * (np.mean(vec_LyX, axis=0) + lambd * theta[k])
        thet0 = theta0[k] - rho * (np.mean(loss01_y))
        
        # append to solution
        theta = np.vstack((theta, thet))
        theta0 = np.append(theta0, thet0)
        
        # iterate
        k += 1
        J_vec = np.append(J_vec, J_loss(theta[k], theta0[k], X, Y, lambd))
        
    # look at optimum
    if abs(J_vec[k+1] - J_vec[k]) > tol:
        print("Couldn't reach optimum")
    else:
        print("Reached optimum at k = {0}".format(k))
    
    return(theta, theta0, k)
        

