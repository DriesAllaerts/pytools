#!/usr/bin/env python

'''
Progress bar for showing progress of time-consuming computations

Author: Dries Allaerts
Date: May 10, 2017
'''

class progressBar(object):
    def __init__(self,text,N):
        print('    '+text)
        print('    1----------------------------------------100')
        print('     ',end='')

        self.__N = N
        self.__counter = 0

    @property
    def N(self):
        return self.__N

    def incr(self,n):
        dn = round(n/self.N*40)
        ni = dn-self.__counter
        for i in range(ni):
            print('|',end='',flush=True)

        self.__counter = dn
        #Reached the end
        if n==self.N:
            print('\n')
