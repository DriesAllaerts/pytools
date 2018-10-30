#!/usr/bin/env python

'''
Some general python functions

- heaviside
- step
- pulse
- smoothstep

Author: Dries Allaerts
Date: May 10, 2017
'''

import numpy as np

def heaviside(x):
    '''
    Heaviside function:
    -------------------
    1.0 for x > 0
    0.5 for x = 0
    0.0 for x < 0
    a tolerance of 1e-12 is allowed
    '''
    return 0.5*(np.sign(np.around(x,12))+1)


def step(x):
    '''
    Step function:
    --------------
    1.0 for x > 0
    0.0 for x <= 0
    a tolerance of 1e-12 is allowed
    '''
    return 1.0 * (np.around(x,12) > 0.0)


def pulse(x,delta):
    '''
    Pulse function:
    ---------------
    1.0 for abs(x) < delta
    0.5 for abs(x) = delta
    0.0 for abs(x) > delta
    '''
    return step(delta-np.abs(x))


def smoothstep(x):
    '''
    Smooth step function:
    ---------------------
    0.0                    for x <= 0
    1/[1+exp(1/(x-1)+1/x)] for 0 < x < 1
    1.0                    for x >= 1
    '''
    with np.errstate(divide='ignore',over='ignore'):
        S = 1.0/(1.0+np.exp(1/(x-1)+1/x)) * ( np.all([x > 0.0,x < 1.0],axis=0) )
    S[x >= 1.0] = 1.0
    return S
