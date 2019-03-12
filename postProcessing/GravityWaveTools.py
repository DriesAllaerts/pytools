#!/usr/bin/env python

'''
Tools for gravity wave postprocessing

Author: Dries Allaerts
Date: March 11, 2019
'''

import numpy as np

def upwardWave(f):
    '''
    Extract part of the wave field that corresponds to
    upward propagating gravity waves, given by quadrants
    I and III in 2D Fourier space. The data points
    should be distributed equidistantly

    Parameters
    ----------
    f: numpy 2D array with shape [Nx, Nz]
        wave field to be decomposed

    Returns
    -------
    fup: numpy 2D array with shape [Nx,Nz]
        upward propagating wave field
    '''
    Nx,Nz = f.shape

    #Create wavenumbers
    #Note: only sign matters, so factor 2pi
    #      and grid spacing are ignored
    ks = np.fft.fftshift(np.fft.fftfreq(Nx))
    ms = np.fft.fftshift(np.fft.fftfreq(Nz))
    Ks, Ms = np.meshgrid(ks,ms,indexing='ij')

    #Convert to Fourier space and extract
    #upward propagating part
    f_c = np.fft.fftshift(np.fft.fft2(f))/f.size
    fup_c = f_c.copy()

    fup_c[(Ks<=0) & (Ms>=0)] = 0.0 #Quadrant II
    fup_c[(Ks>=0) & (Ms<=0)] = 0.0 #Quadrant IV

    fup = np.fft.ifft2(np.fft.ifftshift(fup_c*fup_c.size))

    return np.real(fup)

def downwardWave(f):
    '''
    Extract part of the wave field that corresponds to
    downward propagating gravity waves, given by quadrants
    II and IV in 2D Fourier space. The data points in f
    should be distributed equidistantly

    Parameters
    ----------
    f: numpy 2D array with shape [Nx, Nz]
        wave field to be decomposed

    Returns
    -------
    fdown: numpy 2D array with shape [Nx,Nz]
        downward propagating wave field
    '''
    Nx,Nz = f.shape

    #Create wavenumbers
    #Note: only sign matters, so factor 2pi
    #      and grid spacing are ignored
    ks = np.fft.fftshift(np.fft.fftfreq(Nx))
    ms = np.fft.fftshift(np.fft.fftfreq(Nz))
    Ks, Ms = np.meshgrid(ks,ms,indexing='ij')

    #Convert to Fourier space and extract
    #downward propagating part
    f_c = np.fft.fftshift(np.fft.fft2(f))/f.size
    fdown_c = f_c.copy()

    fdown_c[(Ks>=0) & (Ms>=0)] = 0.0 #Quadrant I
    fdown_c[(Ks<=0) & (Ms<=0)] = 0.0 #Quadrant III

    fdown = np.fft.ifft2(np.fft.ifftshift(fdown_c*fdown_c.size))

    return np.real(fdown)
