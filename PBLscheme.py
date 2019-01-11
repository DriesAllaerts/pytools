#!/usr/bin/env python

'''
Planetary boundary layer schemes
'''

__author__ = "Dries Allaerts"
__date__   = "January 11, 2019"

import numpy as np
import os

def PBLTroenMahrt(z,h,ust,Tst,zeta):
    p = 2
    eps = 0.1
    C = 6.5
    kappa = 0.4
    gravity = 9.81
    T0 = 300.
    
    Nz = z.size
    Nt = ust.size
    Km = np.zeros((Nt,Nz))
    ws = np.zeros((Nt,Nz))
    Kh = np.zeros((Nt,Nz))

    z   = np.tile(z,(Nt,1))
    h   = np.tile(h,(Nz,1)).T
    ust = np.tile(ust,(Nz,1)).T
    Tst = np.tile(Tst,(Nz,1)).T

    #Stable conditions
    ind = (Tst>=0)
    Km[ind] = ust[ind] * kappa * z[ind] / (1+4.7*zeta[ind]) * (1 - z[ind]/h[ind])**p

    #Unstable conditions
    ind = (Tst<0)
    ws[ind] = (ust[ind]**3 - 7*eps*kappa*gravity/T0 * h[ind]*Tst[ind]*ust[ind])**(1./3.)
    Km[ind] = ws[ind] * kappa * z[ind] * (1 - z[ind]/h[ind])**p

    ind = (Tst<0) & (z<eps*h)
    Km[ind] = ust[ind] * kappa * z[ind] * (1-7.*zeta[ind])**(1./3.) * (1 - z[ind]/h[ind])**p



    #Set Km to zero above boundary layer
    ind = (z>h)
    Km[ind] = 0.

    return Km
