# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:49:27 2020
@author: Ryo

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


def svm_gradient(X, Y, rho, lambd, tol, max_iter):
    # fixed values
    #number of observations and number of variables
    p = np.shape(X)[1]
    # tuple of row vectors of the matrix
    X_tuple = tuple(X)
        
    # initialize
    theta0 = np.array(0).reshape(1)
    theta = np.zeros((1,p))
    k = 0
    
    # save data
    J_vec = np.array(0)
    J_vec = np.append(J_vec, J_loss(theta[k], theta0[k], X, Y, lambd) )
    
    while abs(J_vec[k+1] - J_vec[k]) > tol and k < max_iter:
        
        # inner operations
        loss01_y = np.array([yi * loss_01(theta[k], theta0[k], xi, yi) for xi, yi in zip(X_tuple,Y)])
        vec_LyX = (X.T * loss01_y).T
        
        # update
        thet = theta[k] - rho * (np.mean(vec_LyX, axis=0) + lambd * theta[k])
        thet0 = theta0[k] - rho * (np.mean(loss01_y))
        
        # append to solution
        theta = np.vstack((theta, thet))
        # theta = np.concatenate((theta, thet), axis=0)
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
        

# =============================================================================
#   TESTS
# =============================================================================
# np.random.seed(1)
# x_test = np.random.randn(6,2) - 2
# x_test = np.vstack((x_test, np.random.randn(4,2) + 2))

# y_test = np.array([-1 if x < 6 else 1 for x in range(10)])

# theta0_test = np.random.random()
# theta_test = np.random.randn(2)
# lambd_test = 1
# rho_test = 0.1
# tol_test = 1e-3
# max_iter = 100

# # test SVM Gradient (is ok)
# th_ans, th0_ans, k_ans = svm_gradient(x_test, y_test, rho_test, lambd_test, tol_test, max_iter)

# # # look data and SVM
# x = x_test[:,0]
# y = x_test[:,1]
# t = np.linspace(x.min(), x.max())

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(x, y, c=y_test)
# ax.plot(t, -(th_ans[k_ans][0] * t + th0_ans[k_ans]) / th_ans[k_ans][1], color='lightblue')
# plt.show()


















