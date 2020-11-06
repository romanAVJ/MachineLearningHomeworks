# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:19:47 2020

@author: Roman AVJ

Util functions 
"""
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Extra Functions
# =============================================================================

def plot_decision_boundary(model, X, y, typeof=1):

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 0.2, X[0, :].max() + 0.2
    y_min, y_max = X[1, :].min() - 0.2, X[1, :].max() + 0.2
    h = 0.01
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    if typeof == 1:
        # color all the surface
        plt.contourf(xx, yy, Z, cmap='bwr', alpha = 0.3)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y, cmap='bwr', alpha = 0.8)
    else:
        # plot decision boundry
        plt.scatter(X[0, :], X[1, :], c=y, cmap='bwr', alpha=0.8)
        plt.contour(xx, yy, Z, levels=1, colors='g', linewidths=2)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    