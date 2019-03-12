#!/usr/bin/env python

'''
Decompose velocity field in upward and downward propagating
wave fields

Expects structured vtk files in a directory structure:
    postProcessing/surfaces/<time>/<basename>_U.vtk

Run from command line as:
$SOURCEPYTHON3
$srun -n 144 python sowfa_decomposeWaveFields.py
'''
from __future__ import print_function

__author__ = "Dries Allaerts"
__date__   = "March 11, 2019"

import numpy as np
import os
import sys
import datetime
import time
import pandas as pd
from mpi4py import MPI
import argparse
from datatools.vtkTools import read_vtkStructured,write_vtkStructured
from pytools.GravityWaveTools import upwardWave,downwardWave

def main(srcdir,basename,verbose):
    #find time subdirectories
    dirlist = []
    for d in os.listdir(srcdir):
        try: 
            step = float(d) # need this to verify this is a time-step dir!
        except ValueError:
            pass
        else:
            dirlist.append(os.path.join(srcdir,d))
    if len(dirlist) == 0:
        sys.exit('No time subdirectories found in '+str(dirs))
    #determine base name
    if basename=='':
        d = dirlist[0]
        filelist = [ f for f in os.listdir(d)
                        if os.path.isfile(os.path.join(d,f)) ]
        for f in filelist:
            if f.startswith('.'):
                continue
            name,ext = os.path.splitext(f)
            if name.endswith('_U'):
                name = name.replace('_U','')
                basename = name
        assert (not basename==''), 'Error: basename could not be identified'
    
    #Global number of tasks
    nTaskGlobal = len(dirlist)

    #Initialise communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nProc = comm.Get_size()
    #Divide tasks over the processors
    nTask = int(nTaskGlobal/nProc)+1*(rank<nTaskGlobal%nProc)
    offset = rank*nTask+(nTaskGlobal%nProc)*(rank>=nTaskGlobal%nProc)
    
    #Perform assigned number of tasks
    for task in range(nTask):
        #Print progress
        if verbose:
            print( 'Task {globalTaskNumber:0>5}/{nTaskGlobal:0>6} started by rank {ranknr:0>4} ({taskNumber:0>3}/{nTask:0>3}) at'.format(
                        globalTaskNumber=task+offset+1,
                        ranknr=rank,
                        taskNumber=task+1,
                        nTask=nTask,
                        nTaskGlobal=nTaskGlobal),str(datetime.datetime.now()),flush=True)

        #Do computation
        d = dirlist[task+offset]
        #Read data
        data, meta = read_vtkStructured(os.path.join(d,basename+'_U.vtk'))
        [X,Y,Z,U,V,W] = data
        
        #Extract upward propagating wave part
        Uup = upwardWave(U.squeeze())
        Vup = upwardWave(V.squeeze())
        Wup = upwardWave(W.squeeze())
        #Assemble data
        Uup = np.expand_dims(Uup,axis=1)
        Vup = np.expand_dims(Vup,axis=1)
        Wup = np.expand_dims(Wup,axis=1)
        dataup = [X,Y,Z,Uup,Vup,Wup]
        write_vtkStructured(dataup,meta,os.path.join(d,basename+'_Uup.vtk'),basename)
        
        #Extract downward propagating wave part
        Udown = downwardWave(U.squeeze())
        Vdown = downwardWave(V.squeeze())
        Wdown = downwardWave(W.squeeze())
        #Assemble data
        Udown = np.expand_dims(Udown,axis=1)
        Vdown = np.expand_dims(Vdown,axis=1)
        Wdown = np.expand_dims(Wdown,axis=1)
        datadown = [X,Y,Z,Udown,Vdown,Wdown]
        write_vtkStructured(datadown,meta,os.path.join(d,basename+'_Udown.vtk'),basename)

    return 0

if __name__ == '__main__':
    # Specify optional arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('srcdir',nargs='?',default=os.getcwd(),help='alternate source directory')
    arg_parser.add_argument('--basename',nargs=1,dest='basename',default='',help='file base name')
    arg_parser.add_argument('--noverbose',dest='verbose',help='reduce output verbosity',action='store_false')
    args = arg_parser.parse_args()

    # Execute main function
    returnFlag = main(args.srcdir,args.basename,args.verbose)

    # Exit
    sys.exit(returnFlag)
