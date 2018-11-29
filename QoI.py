#!/usr/bin/env python

'''
Functions to compute quantities of interest
- REWS:  rotor equivalent wind speed
- alpha: wind shear exponent
- psi:   wind veer

Author: Dries Allaerts
Date: May 10, 2017
'''

import numpy as np
from scipy.interpolate import interp1d
from scipy import optimize


def REWS(z,S,WD,zh,D,ax=-1):
    z1 = zh - D/2.
    z2 = zh + D/2.
    zcc,dz = np.linspace(z1,z2,10,retstep=True)
    zst = np.linspace(z1-dz/2.,z2+dz/2.,11)
    dareas = disk_areas(zst,zh,D/2.)

    A = np.pi*D**2./4.

    #Move specified ax to last position, then reshape to 2d array and iterate
    Nz = S.shape[ax]
    N  = int(S.size/Nz)
    new_shape = np.moveaxis(S,ax,-1).shape[:-1]
    REWS = np.zeros((N))
    for i in range(N):
        Sint = interp1d(z,np.moveaxis(S,ax,-1).reshape(N,Nz)[i,:],fill_value='extrapolate')(zcc)
        WDint = interp1d(z,np.moveaxis(WD,ax,-1).reshape(N,Nz)[i,:],fill_value='extrapolate')(zcc)
        WDhub = interp1d(z,np.moveaxis(WD,ax,-1).reshape(N,Nz)[i,:],fill_value='extrapolate')(zh)
        REWS[i] = (1./A * np.sum( dareas * Sint**3 * np.cos(np.radians(WDint-WDhub)) ) )**(1./3.)

    return REWS.reshape(new_shape)


def alpha(z,S,zh,D,ax=-1):
    z1 = zh - D/2.
    z2 = zh + D/2.
    zcc = np.linspace(z1,z2,10)

    #Move specified ax to last position, then reshape to 2d array and iterate
    Nz = S.shape[ax]
    N  = int(S.size/Nz)
    new_shape = np.moveaxis(S,ax,-1).shape[:-1]
    alpha = np.zeros((N))
    for i in range(N):
        Sint = interp1d(z,np.moveaxis(S,ax,-1).reshape(N,Nz)[i,:],fill_value='extrapolate')(zcc)
        Shub = interp1d(z,np.moveaxis(S,ax,-1).reshape(N,Nz)[i,:],fill_value='extrapolate')(zh)

        f = lambda x, alpha: Shub*(x/zh)**alpha
        popt,_ = optimize.curve_fit(f,zcc,Sint,1.0)
        alpha[i] = popt[0]
    return alpha.reshape(new_shape)


def psi(z,WD,zh,D,ax=-1):
    z1 = zh - D/2.
    z2 = zh + D/2.
    zcc = np.linspace(z1,z2,10)

    #Move specified ax to last position, then reshape to 2d array and iterate
    Nz = WD.shape[ax]
    N  = int(WD.size/Nz)
    new_shape = np.moveaxis(WD,ax,-1).shape[:-1]
    psi = np.zeros((N))
    for i in range(N):
        WDint = interp1d(z,np.moveaxis(WD,ax,-1).reshape(N,Nz)[i,:],fill_value='extrapolate')(zcc)
        WDhub = interp1d(z,np.moveaxis(WD,ax,-1).reshape(N,Nz)[i,:],fill_value='extrapolate')(zh)

        f = lambda x, *p: p[0]*x + p[1]
        popt,_ = optimize.curve_fit(f,zcc-zh,WDint-WDhub,[1.0,0.0])
        psi[i] = popt[0]
    return psi.reshape(new_shape)


def disk_areas(zst,zc,r):
    '''
    For a circle with given height and radius, find the area of the circle
    segment at every grid cell of a given vertical grid

    Parameters
    ----------
    zst: numpy 1D array of Nz+1
        faces of vertical grid cells (=staggered grid)
    zc,r: float
        center and radius of the circle

    Returns (integral)
    ------------------
    areas: numpy 1D array
        areas of circle sigments
    '''
    Nz = zst.shape[0]-1
    areas = np.zeros((Nz))
    for k in range(Nz):
        zl = zst[k]
        zh = zst[k+1]
        if (zl<=zc-r) and np.abs(zh-zc)<r:
            theta = 2*np.arccos(np.abs(zc-zh)/r)
            areas[k] = 0.5*r**2*(theta-np.sin(theta))
        elif np.abs(zh-zc)<r and np.abs(zl-zc)<r:
            if zh<=zc or zl>=zc:
                theta1 = 2*np.arccos(np.abs(zc-zh)/r)
                area1  = 0.5*r**2*(theta1-np.sin(theta1))
                theta2 = 2*np.arccos(np.abs(zc-zl)/r)
                area2  = 0.5*r**2*(theta2-np.sin(theta2))
                areas[k] = np.abs(area1-area2)
            elif zl<zc and zh>zc:
                theta1 = 2*np.arccos(np.abs(zc-zh)/r)
                area1  = 0.5*r**2*(theta1-np.sin(theta1))
                theta2 = 2*np.arccos(np.abs(zc-zl)/r) 
                area2  = 0.5*r**2*(theta2-np.sin(theta2))
                areas[k] = np.pi*r**2-area1-area2
            else:
                print('Error, you should not end up here')
                return 1
        elif (zh>=zc+r) and np.abs(zl-zc)<r:
            theta = 2*np.arccos(np.abs(zc-zl)/r)
            areas[k] = 0.5*r**2*(theta-np.sin(theta))
        #else: point lies outside the circle so do not
    return areas


def filter1d(a,b,ax=-1):
    '''
    Multiplicate ndarray a with 1d array b along given axis
    '''
    
    # Create an array which would be used to reshape 1D array, b to have 
    # singleton dimensions except for the given axis where we would put -1 
    # signifying to use the entire length of elements along that axis  
    dim_array = np.ones((1,a.ndim),int).ravel()
    dim_array[ax] = -1
    
    # Reshape b with dim_array and perform elementwise multiplication with 
    # broadcasting along the singleton dimensions for the final output
    b_reshaped = b.reshape(dim_array)
    return a*b_reshaped 
