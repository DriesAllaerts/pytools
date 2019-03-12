#!/usr/bin/env python

'''
Routines to estimate the height of the planetary boundary layer
'''

__author__ = "Dries Allaerts"
__date__   = "January 11, 2019"

import numpy as np
import os

def modifiedRiMethod(z,             #Vertical grid levels
                     U,             #Wind speed
                     T,             #Potential temperature
                     Tskin,         #Surface skin potential temperature
                     ust,           #Friction velocity
                     Tst,           #Temperature scale
                     gravity=9.81,  #Gravitational acceleration
                     T0=300.0,      #Reference temperature
                     C=6.5,         #Countergradient flux constant
                     eps=0.1,       #Ratio of surface layer to boundary-layer height
                     kappa=0.4,     #von Karman constant
                     Ric=0.5        #Critical Richardson number
                     ):
    '''
    Determine the boundary-layer depth based on
    a modified bulk Richardson number

    see Troen and Mahrt (1986)
    '''
    #First nonzero vertical level
    k1 = np.min(np.nonzero(z))

    def convectiveVelocityScale(h,ust0,Tst0):
        wst3 = -gravity*h/T0 * Tst0 * ust0
        ws = (ust0**3 + 7.*eps*kappa*wst3)**(1./3.)
        return ws

    Nt = ust.size
    Nz = z.size
    h = np.zeros((Nt))
    for t in range(Nt):
        k = k1
        Ri0 = 0.0
        while (k < Nz):
            if Tst[t] <= 0:
                ws = convectiveVelocityScale(z[k],ust[t],Tst[t])
                Ts = T[t,k1] - C * ust[t] * Tst[t] / ws
            else:
                Ts = Tskin[t]
            
            Ri = gravity*z[k] * (T[t,k] - Ts) / (T0 * U[t,k]**2)
            if Ri > Ric:
                h[t] = z[k] + (z[k] - z[k-1])/(Ri-Ri0)*(Ric-Ri)
                break
            
            Ri0 = Ri
            k += 1


    return h
