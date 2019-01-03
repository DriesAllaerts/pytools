#!/usr/bin/env python

'''
Functions for curve fitting
- weighted least squares

Author: Dries Allaerts
Date: January 3, 2019
'''

import numpy as np
from scipy.linalg import inv

def weightedLeastSquares(xdata,ydata,N,weights=np.array([])):
    '''
    Find the (weighted) polynomial fit of
    degree N through the data points (x,y)
    - N = 0: weighted mean
    - N = 1: weighted linear regression
    - N = 2: weighted quadratic regression
    - N = 3: weighted cubic regression
    - ...

    The weights of the polynomial are given by
    beta = inv(X.T W X) X.T W y
    with X[i,:] = [1 x[i] x[i]**2 ... x[i]**N]
    and  W[i,i] = w[i]
    '''
    
    #Check that x and y have compatible dimensions
    assert (len(xdata.shape)==1), \
        'Error: x should be a vector'
    assert (xdata.size == ydata.shape[0]), \
        'Error: first dimension of y should equal length of x'
    assert (N>=0), \
        'Error: N should be positive'

    X = np.stack( [xdata**i for i in range(N+1)] ).T

    if weights.size==0:
        W = np.eye(xdata.size)
    else:
        assert (weights.size==xdata.size), \
            'Error: weights and xdata should be of the same size'

        W = np.diag(weights)

    XtW = np.matmul(X.T,W)
    XtWX = np.matmul(XtW,X)
    XtWXinv = inv(XtWX)
    beta = np.matmul( np.matmul(XtWXinv,XtW), ydata)
    return beta
