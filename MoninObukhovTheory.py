#!/usr/bin/env python

'''
Monin-Obukhov similarity theory

By convention, the symbol T is used to represent potential temperature
'''

__author__ = "Dries Allaerts"
__date__   = "January 9, 2019"

import numpy as np
import os

class MOST(object):
    
    def __init__(self,z0,**kwargs):
        #Surface roughness height
        self.__z0 = z0
        
        #Reference temperature
        try:
            self.__T0 = kwargs['T0']
        except KeyError:
            self.__T0 = 300.0

        #Von Karman constant
        try:
            self.__kappa = kwargs['kappa']
        except KeyError:
            self.__kappa = 0.41

        #Gravitational acceleration
        try:
            self.__gravity = kwargs['gravity']
        except KeyError:
            self.__gravity = 9.81

        #MO parameters
        try:
            self.__betam = kwargs['betam']
        except KeyError:
            self.__betam = 4.8

        try:
            self.__betah = kwargs['betah']
        except KeyError:
            self.__betah = 7.8

        try:
            self.__gammam = kwargs['gammam']
        except KeyError:
            self.__gammam = 19.3

        try:
            self.__gammah = kwargs['gammah']
        except KeyError:
            self.__gammah = 11.6

        #Numerical parameters
        try:
            self.__tolerance = kwargs['tolerance']
        except KeyError:
            self.__tolerance = 1.0e-10

        try:
            self.__maxCount = kwargs['maxCount']
        except KeyError:
            self.__maxCount = 100

        try:
            self.__alpha = kwargs['alpha']
        except KeyError:
            self.__alpha = 1.0

#    def setMOParameters():

    def calculateFluxes(self,z,U,**kwargs):
        assert (z > self.z0), \
            'Error: z should be larger than the surface roughness'

        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = False
        
        if 'Ts' in kwargs and 'T' in kwargs:
            Ts = kwargs['Ts']
            T  = kwargs['T']
        elif 'T2' in kwargs and 'T' in kwargs:
            T2 = kwargs['T2']
            T  = kwargs['T']
        elif 'qw' in kwargs and not 'T' in kwargs:
            qw = kwargs['qw']
        else:
            print('Error: Either temperatures "T" and "Ts" or "T2", or surface heat flux "qw" must be specified')
            return 1

        #initial guess for ust, Tst and zeta
        ust = self.ust(z,U)
        if 'T' in kwargs:
            if 'Ts' in kwargs:
                Tst = self.Tst(z,T,Ts=kwargs['Ts'])
            else:
                Tst = self.Tst(z,T,T2=kwargs['T2'])
            zeta0 = self.zeta(z,ust,Tst=Tst)
        else:
            zeta0 = self.zeta(z,ust,qw=qw)
        
        # Loop until convergence
        count = 0
        error = 1
        while (error > self.tolerance and count < self.maxCount):
            ust = self.ust(z,U,zeta=zeta0)
            if 'T' in kwargs:
                if 'Ts' in kwargs:
                    Tst = self.Tst(z,T,Ts=kwargs['Ts'],zeta=zeta0)
                else:
                    Tst = self.Tst(z,T,T2=kwargs['T2'],zeta=zeta0)
                zeta = (1-self.alpha)*zeta0 + self.alpha*self.zeta(z,ust,Tst=Tst)
            else:
                zeta = (1-self.alpha)*zeta0 + self.alpha*self.zeta(z,ust,qw=qw)
                Tst = -qw/ust

            error = np.max(np.abs(zeta - zeta0))
            zeta0 = zeta
            count += 1
            if verbose:
                print('Iteration',count,': residual =',error)

        if count == self.maxCount:
            print('Warning: performed maximum number of iterations without reaching convergence')

        if np.isscalar(U):
            ust = np.asscalar(ust)
            Tst = np.asscalar(Tst)
        return ust,Tst


    def ust(self,z,U,**kwargs):
        try:
            zeta = kwargs['zeta']
        except KeyError:
            zeta = np.array([0.0,])

        return self.kappa * U / ( np.log(z/self.z0) - self.Psim(zeta) + self.Psim(self.z0/z*zeta) )


    def Tst(self,z,T,**kwargs):
        try:
            zeta = kwargs['zeta']
        except KeyError:
            zeta = np.array([0.0,])

        if 'Ts' in kwargs:
            return self.kappa * (T - kwargs['Ts']) / ( np.log(z/self.z0) - self.Psih(zeta) + self.Psih(self.z0/z*zeta) )
        elif 'T2' in kwargs:
            return self.kappa * (T - kwargs['T2']) / ( np.log(z/2.) - self.Psih(zeta) + self.Psih(2./z*zeta) )
        else:
            print('Error: Either temperatures "T" and "Ts" or "T2", or surface heat flux "qw" must be specified')
            return 1


    def zeta(self,z,ust,**kwargs):
#        assert (np.isscalar(z)), \
#            'Error: z should be a scalar'
#        assert (type(ust) == type(Tst)), \
#            'Error: ust and Tst should be of the same type'
#        if not np.isscalar(ust):
#            assert (type(ust) == np.ndarray)
#            assert (ust.size == Tst.size), \
#                'Error: ust and Tst should have the same size'
        #if z is an array, output has shape (ust.size,z.size)
        if np.isscalar(z):   z = np.array([z,])
        if np.isscalar(ust): ust = np.array([ust,])
        Nz = z.size
        Nt = ust.size
        z   = np.tile(z,(Nt,1))
        ust = np.tile(ust,(Nz,1)).T

        if 'Tst' in kwargs:
            Tst = kwargs['Tst']
            if np.isscalar(Tst): Tst = np.array([Tst,])
            Tst = np.tile(Tst,(Nz,1)).T
            return np.squeeze( self.kappa * self.gravity * z * Tst / (ust**2 * self.T0) )
        elif 'qw' in kwargs:
            qw = kwargs['qw']
            if np.isscalar(qw): qw = np.array([qw,])
            qw = np.tile(qw,(Nz,1)).T
            return np.squeeze( -self.kappa * self.gravity * z * qw / (ust**3 * self.T0) )
        else:
            print('Error: Either temperature scale "Tst" or heat flux "qw" must be specified')
            return 1


    def Psim(self,zeta):
        if np.isscalar(zeta): zeta = np.array([zeta,])
        Psim = np.zeros(zeta.shape)

        Psim[zeta<0.]  = self.PsimUnstable(zeta[zeta<0.])
        Psim[zeta>=0.] = self.PsimStable(zeta[zeta>=0.])
        return Psim


    def Psih(self,zeta):
        if np.isscalar(zeta): zeta = np.array([zeta,])
        Psih = np.zeros(zeta.shape)

        Psih[zeta<0.]  = self.PsihUnstable(zeta[zeta<0.])
        Psih[zeta>=0.] = self.PsihStable(zeta[zeta>=0.])
        return Psih


    def PsimStable(self,zeta):
        return -self.betam*zeta


    def PsihStable(self,zeta):
        return -self.betah*zeta


    def PsimUnstable(self,zeta):
        phim = self.phimUnstable(zeta)
        Psim = np.log( ( (1 + phim**(-2.))/2. ) * ( (1 + phim**(-1.))/2. )**2. ) \
                - 2 * np.arctan( phim**(-1.) ) + np.pi/2.
        return Psim


    def PsihUnstable(self,zeta):
        phih = self.phihUnstable(zeta)
        Psih = np.log( ( (1 + phih**(-1.))/2. )**2. )
        return Psih


    def phim(self,zeta):
        if np.isscalar(zeta): zeta = np.array([zeta,])
        phim = np.zeros(zeta.shape)

        phim[zeta<0.]  = self.phimUnstable(zeta[zeta<0.])
        phim[zeta>=0.] = self.phimStable(zeta[zeta>=0.])
        return phim


    def phih(self,zeta):
        if np.isscalar(zeta): zeta = np.array([zeta,])
        phih = np.zeros(zeta.shape)

        phih[zeta<0.]  = self.phihUnstable(zeta[zeta<0.])
        phih[zeta>=0.] = self.phihStable(zeta[zeta>=0.])
        return phih


    def phimStable(self,zeta):
        return 1 + self.betam * zeta


    def phihStable(self,zeta):
        return 1 + self.betah * zeta


    def phimUnstable(self,zeta):
        return (1 - self.gammam*zeta)**(-0.25)


    def phihUnstable(self,zeta):
        return (1 - self.gammah*zeta)**(-0.5)


    @property
    def z0(self):
        return self.__z0
    @property
    def T0(self):
        return self.__T0
    @property
    def kappa(self):
        return self.__kappa
    @property
    def gravity(self):
        return self.__gravity
    @property
    def betam(self):
        return self.__betam
    @betam.setter
    def betam(self,value):
        self.__betam = value
    @property
    def betah(self):
        return self.__betah
    @betah.setter
    def betah(self,value):
        self.__betah = value
    @property
    def gammam(self):
        return self.__gammam
    @gammam.setter
    def gammam(self,value):
        self.__gammam = value
    @property
    def gammah(self):
        return self.__gammah
    @gammah.setter
    def gammah(self,value):
        self.__gammah = value
    @property
    def tolerance(self):
        return self.__tolerance
    @tolerance.setter
    def tolerance(self,value):
        self.__tolerance = value
    @property
    def maxCount(self):
        return self.__maxCount
    @maxCount.setter
    def maxCount(self,value):
        self.__maxCount = int(value)
    @property
    def alpha(self):
        return self.__alpha
    @alpha.setter
    def alpha(self,value):
        self.__alpha = value
