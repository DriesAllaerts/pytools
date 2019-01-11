#!/usr/bin/env python

'''
Surface-layer schemes

By convention, the symbol T is used to represent potential temperature
'''

__author__ = "Dries Allaerts"
__date__   = "January 9, 2019"

import numpy as np
import os

class MOST(object):
    '''
    Monin-Obukhov similarity theory

    Implentation follows section 2 of
    Blumel (2000) An approximate analytical solution of
    flux-profile relationships for the atmospheric surface
    layer with different momentum and heat roughness lengths
    '''
    def __init__(self,z0,**kwargs):
        '''
        Default values

        ABL parameters
        - Surface roughness height for heat     z0t = z0
        - Reference temperature                 T0  = 300 K
        - Gravitational acceleration            g   = 9.81 m/s^2
        - useT0 = False (unless T0 is given or specified directly)

        MO parameters
        - kappa   = 0.41    (von Karman constant)
        - a_h     = 1       (inverse of turbulent Prandtl number at neutral conditions)
        - beta_m  = 4.8
        - beta_h  = 7.8
        - gamma_m = 19.3
        - gamma_h = 11.6
        - veryStableRegime = True

        Numerical parameters
        - Maximum number of iterations      maxCount  = 100
        - Tolerance                         tolerance = 1.0e-10
        - Relaxation coefficient            alpha     = 1.0
        '''
        #Surface roughness height
        self.__z0 = z0

        #Surface roughness height for heat
        try:
            self.__z0t = kwargs['z0t']
        except KeyError:
            self.__z0t = self.z0

        #Reference temperature
        try:
            self.__T0 = kwargs['T0']
            self.__useT0 = True
        except KeyError:
            self.__T0 = 300.0
            self.__useT0 = False

        try:
            self.__useT0 = kwargs['useT0']
        except KeyError:
            pass

        #Gravitational acceleration
        try:
            self.__gravity = kwargs['gravity']
        except KeyError:
            self.__gravity = 9.81


        #MO parameters

        #von Karman constant
        try:
            self.__kappa = kwargs['kappa']
        except KeyError:
            self.__kappa = 0.41

        #inverse of turbulent Prandtl number at neutral conditions
        try:
            self.__ah = kwargs['ah']
        except KeyError:
            self.__ah = 1.


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

        try:
            self.__veryStableRegime = kwargs['veryStableRegime']
        except KeyError:
            self.__veryStableRegime = True

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

    def setMOParameters(self,setname='default'):
        if setname == 'Brutsaert1982':
            print('Using values of Brutsaert (1982)')
            self.kappa  = 0.4
            self.ah     = 1.0
            self.betam  = 5.0
            self.betah  = 5.0
            self.gammam = 16.0
            self.gammah = 16.0
        elif setname == 'Businger1971':
            print('Using values of Businger (1971)')
            self.kappa  = 0.35
            self.ah     = 1.35
            self.betam  = 4.7
            self.betah  = 6.35
            self.gammam = 15.0
            self.gammah = 9.0
        else: #revert to default values
            print('Using default values')
            self.kappa  = 0.41
            self.ah     = 1.0
            self.betam  = 4.8
            self.betah  = 7.8
            self.gammam = 19.3
            self.gammah = 11.6
        return

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
            Tinput = 'Ts'
        elif 'T2' in kwargs and 'T' in kwargs:
            T2 = kwargs['T2']
            T  = kwargs['T']
            Tinput = 'T2'
        elif 'qw' in kwargs:
            qw = kwargs['qw']
            Tinput = 'qw'
        else:
            print('Error: Either temperatures "T" and "Ts" or "T2", or surface heat flux "qw" must be specified')
            return 1
        
        # Set reference temeprature to be used in the computation of zeta
        # (i.e. in the Obukhov length). If useT0 is True but T is not specified
        # (e.g. when only qw is specified), revert to using T0
        if not self.useT0:
            try:
                TRef = kwargs['T']
            except KeyError:
                print('Warning: T is not provided, so using T0 to compute Obukhov length')
                TRef = self.T0
        else:
            TRef = self.T0

        #initial guess for ust, Tst and zeta
        ust = self.ust(z,U)
        if Tinput == 'Ts':
            Tst = self.Tst(z,T,Ts=Ts)
            zeta0 = self.zeta(z,ust,Tst=Tst,TRef=TRef)
        elif Tinput == 'T2':
            Tst = self.Tst(z,T,T2=T2)
            zeta0 = self.zeta(z,ust,Tst=Tst,TRef=TRef)
        else:
            zeta0 = self.zeta(z,ust,qw=qw,TRef=TRef)
        
        # Loop until convergence
        count = 0
        error = 1
        while (error > self.tolerance and count < self.maxCount):
            ust = self.ust(z,U,zeta=zeta0)
            if Tinput == 'Ts':
                Tst = self.Tst(z,T,Ts=Ts,zeta=zeta0)
                zeta = (1-self.alpha)*zeta0 + self.alpha*self.zeta(z,ust,Tst=Tst,TRef=TRef)
            elif Tinput == 'T2':
                Tst = self.Tst(z,T,T2=T2,zeta=zeta0)
                zeta = (1-self.alpha)*zeta0 + self.alpha*self.zeta(z,ust,Tst=Tst,TRef=TRef)
            else:
                zeta = (1-self.alpha)*zeta0 + self.alpha*self.zeta(z,ust,qw=qw,TRef=TRef)
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
            Tst = self.ah * self.kappa * (T - kwargs['Ts']) / \
                    ( np.log(z/self.z0t) - self.Psih(zeta) + self.Psih(self.z0t/z*zeta) )
            return Tst
        elif 'T2' in kwargs:
            Tst = self.ah * self.kappa * (T - kwargs['T2']) / \
                    ( np.log(z/2.) - self.Psih(zeta) + self.Psih(2./z*zeta) )
            return Tst
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

        #If TRef is not specified, use T0
        if 'TRef' in kwargs:
            TRef = kwargs['TRef']
        else:
            TRef = self.T0
        #Revert TRef to array (Nt,Nz)
        if np.isscalar(TRef): TRef = np.array([TRef,])
        TRef = np.tile(TRef,(Nz,1)).T


        if 'Tst' in kwargs:
            Tst = kwargs['Tst']
            if np.isscalar(Tst): Tst = np.array([Tst,])
            Tst = np.tile(Tst,(Nz,1)).T
            return np.squeeze( self.kappa * self.gravity * z * Tst / (ust**2 * TRef) )
        elif 'qw' in kwargs:
            qw = kwargs['qw']
            if np.isscalar(qw): qw = np.array([qw,])
            qw = np.tile(qw,(Nz,1)).T
            return np.squeeze( -self.kappa * self.gravity * z * qw / (ust**3 * TRef) )
        else:
            print('Error: Either temperature scale "Tst" or heat flux "qw" must be specified')
            return 1


    def Psim(self,zeta):
        if np.isscalar(zeta): zeta = np.array([zeta,])
        Psim = np.zeros(zeta.shape)

        Psim[zeta<0.]  = self.PsimUnstable(zeta[zeta<0.])
        Psim[zeta>=0.] = self.PsimStable(zeta[zeta>=0.])
        if self.veryStableRegime:
            Psim[zeta>=1.] = self.PsimVeryStable(zeta[zeta>=1.])
        return Psim


    def Psih(self,zeta):
        if np.isscalar(zeta): zeta = np.array([zeta,])
        Psih = np.zeros(zeta.shape)

        Psih[zeta<0.]  = self.PsihUnstable(zeta[zeta<0.])
        Psih[zeta>=0.] = self.PsihStable(zeta[zeta>=0.])
        if self.veryStableRegime:
            Psih[zeta>=1.] = self.PsihVeryStable(zeta[zeta>=1.])
        return Psih


    def PsimStable(self,zeta):
        return -self.betam*zeta


    def PsimVeryStable(self,zeta):
        return -self.betam*np.log(zeta) - self.betam


    def PsihStable(self,zeta):
        return -self.betah*zeta


    def PsihVeryStable(self,zeta):
        return -self.betah*np.log(zeta) - self.betah


    def PsimUnstable(self,zeta):
        xm = self.phimUnstable(zeta)
        Psim = np.log( ( (1 + xm**(-2.))/2. ) * ( (1 + xm**(-1.))/2. )**2. ) \
                - 2 * np.arctan( xm**(-1.) ) + np.pi/2.
        return Psim


    def PsihUnstable(self,zeta):
        xh = self.ah * self.phihUnstable(zeta)
        Psih = np.log( ( (1 + xh**(-1.))/2. )**2. )
        return Psih


    def phim(self,zeta):
        if np.isscalar(zeta): zeta = np.array([zeta,])
        phim = np.zeros(zeta.shape)

        phim[zeta<0.]  = self.phimUnstable(zeta[zeta<0.])
        phim[zeta>=0.] = self.phimStable(zeta[zeta>=0.])
        if self.veryStableRegime:
            phim[zeta>=1.] = self.phimVeryStable(zeta[zeta>=1.])
        return phim


    def phih(self,zeta):
        if np.isscalar(zeta): zeta = np.array([zeta,])
        phih = np.zeros(zeta.shape)

        phih[zeta<0.]  = self.phihUnstable(zeta[zeta<0.])
        phih[zeta>=0.] = self.phihStable(zeta[zeta>=0.])
        if self.veryStableRegime:
            phih[zeta>=1.] = self.phihVeryStable(zeta[zeta>=1.])
        return phih


    def phimStable(self,zeta):
        return 1 + self.betam * zeta


    def phimVeryStable(self,zeta):
        return 1 + self.betam


    def phihStable(self,zeta):
        return 1./self.ah * (1 + self.betah * zeta)


    def phihVeryStable(self,zeta):
        return 1./self.ah * (1 + self.betah)


    def phimUnstable(self,zeta):
        return (1 - self.gammam*zeta)**(-0.25)


    def phihUnstable(self,zeta):
        return 1./self.ah * (1 - self.gammah*zeta)**(-0.5)


    @property
    def z0(self):
        return self.__z0
    @property
    def z0t(self):
        return self.__z0t
    @property
    def T0(self):
        return self.__T0
    @property
    def useT0(self):
        return self.__useT0
    @useT0.setter
    def useT0(self,value):
        self._useT0 = bool(value)
    @property
    def gravity(self):
        return self.__gravity
    @property
    def kappa(self):
        return self.__kappa
    @kappa.setter
    def kappa(self,value):
        self.__kappa = value
    @property
    def ah(self):
        return self.__ah
    @ah.setter
    def ah(self,value):
        self.__ah = value
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
    def veryStableRegime(self):
        return self.__veryStableRegime
    @veryStableRegime.setter
    def veryStableRegime(self,value):
        self.__veryStableRegime = bool(value)
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
