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

def low_pass_filter(t,u,Tf):
    '''
    Low pass filter

    Parameters
    ----------
    t: numpy array
        sampling time
    u: numpy array
        signal samples
    Tf: float
        Filter time scale

    Returns
    -------
    uf: numpy array
        low-pass filtered signal
    '''
    N = t.size
    uf = u.copy()
    for i in range(1,N):
        dt = t[i]-t[i-1]
        alpha = Tf/(Tf+dt)
        uf[i] = alpha*uf[i-1] + (1-alpha)*u[i]
    return uf


def high_pass_filter(t,u,Tf):
    '''
    High pass filter

    Parameters
    ----------
    t: numpy array
        sampling time
    u: numpy array
        signal samples
    Tf: float
        Filter time scale

    Returns
    -------
    uf: numpy array
        high-pass filtered signal
    '''
    N = t.size
    uf = u
    for i in range(1,N):
        dt = t[i]-t[i-1]
        alpha = Tf/(Tf+dt)
        uf[i] = alpha*uf[i-1] + alpha*(u[i]-u[i-1])
    return uf

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
