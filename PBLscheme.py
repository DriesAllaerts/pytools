#!/usr/bin/env python

'''
Planetary boundary layer schemes
'''

__author__ = "Dries Allaerts"
__date__   = "January 11, 2019"

import numpy as np
import os
from pytools.SurfaceLayerSchemes import MOST

class TroenMahrtScheme():
    
    def __init__(self,**kwargs):
        #Exponent p
        try:
            self.__p = kwargs['p']
        except KeyError:
            self.__p = 2

        #Ratio of surface layer to boundary-layer height
        try:
            self.__eps = kwargs['eps']
        except KeyError:
            self.__eps = 0.1

        #Countergradient flux constant
        try:
            self.__C = kwargs['C']
        except KeyError:
            self.__C = 6.5

        #Reference temperature
        try:
            self.__T0 = kwargs['T0']
        except KeyError:
            self.__T0 = 300.0

        #Gravitational acceleration
        try:
            self.__gravity = kwargs['gravity']
        except KeyError:
            self.__gravity = 9.81

        #von Karman constant
        try:
            self.__kappa = kwargs['kappa']
        except KeyError:
            self.__kappa = 0.4


    def calculateViscosity(self,z,h,ust,Tst):
        Nz = z.size
        Nt = ust.size
    
        z   = np.tile(z,(Nt,1))
        h   = np.tile(h,(Nz,1)).T
        ust = np.tile(ust,(Nz,1)).T
        Tst = np.tile(Tst,(Nz,1)).T

        Km = self.Km(z,h,ust,Tst)
        Pr = self.Prandtl(h,ust,Tst)
        gamma = self.gamma(h,ust,Tst)

        return Km,Pr,gamma


    def zeta(self,z,ust,Tst):
        return self.kappa * self.gravity * z * Tst / (ust**2 * self.T0)


    def convectiveVelocityScale(self,h,ust,Tst):
        ws = (ust**3 - 7*self.eps*self.kappa*self.gravity/self.T0 * h*Tst*ust)**(1./3.)
        return ws

    def phimStable(self,zeta):
        return 1 + 4.7 * zeta

    def phimUnstable(self,zeta):
        return (1 - 7.*zeta)**(-1./3.)
        

    def Km(self,z,h,ust,Tst):
        Km = np.zeros(z.shape)
        
        zeta = self.zeta(z,ust,Tst)

        #Stable conditions
        ind = (Tst>=0)
        phim = self.phimStable(zeta[ind])
        Km[ind] = ust[ind] * self.kappa * z[ind] / phim * (1 - z[ind]/h[ind])**self.p
    
        #Unstable conditions
        ind = (Tst<0)
        ws = self.convectiveVelocityScale(h[ind],ust[ind],Tst[ind])
        Km[ind] = ws * self.kappa * z[ind] * (1 - z[ind]/h[ind])**self.p
    
        ind = (Tst<0) & (z<self.eps*h)
        phim = self.phimUnstable(zeta[ind])
        Km[ind] = ust[ind] * self.kappa * z[ind] / phim * (1 - z[ind]/h[ind])**self.p
    
        #Set Km to zero above boundary layer
        ind = (z>h)
        Km[ind] = 0.
    
        return Km


    def gamma(self,h,ust,Tst):
        gamma = np.zeros(h.shape)

        #Unstable conditions
        ind = (Tst<0)
        ws = self.convectiveVelocityScale(h[ind],ust[ind],Tst[ind])
        gamma[ind] = - self.C * Tst[ind] * ust[ind] / (ws * h[ind])
        return gamma

    def Prandtl(self,h,ust,Tst):
        #Create MOST model to use phim and phih functions
        #phim and phih are independent of z0, so value of z0 is irrelevant here
        MO = MOST(z0=0.1,T0=self.T0,gravity=self.gravity)
        MO.setMOParameters('Businger1971')
        MO.kappa = self.kappa
        MO.veryStableRegime = False

        zeta = self.zeta(self.eps*h,ust,Tst)

        phim = MO.phim(zeta)
        phih = MO.phih(zeta)

        Pr = (phih/phim + self.kappa * self.eps * self.C)**(-1.)
        return Pr

    @property
    def p(self):
        return self.__p
    @property
    def eps(self):
        return self.__eps
    @property
    def C(self):
        return self.__C
    @property
    def T0(self):
        return self.__T0
    @property
    def gravity(self):
        return self.__gravity
    @property
    def kappa(self):
        return self.__kappa

