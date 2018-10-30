#!/usr/bin/env python

'''
Template for parallel python script

Commands to run script:
$ sourcePYTHON3

$ mpirun -np 16 python templatePythonScriptParallel.py
'''

__author__ = "Dries Allaerts"
__date__   = "October 8, 2018"

import numpy as np
import os
import sys
import datetime
import time
import pandas as pd
from mpi4py import MPI
import argparse

def main(verbose):
    #Global number of tasks
    nTaskGlobal = 20 #REPLACE WITH TOTAL NUMBER OF TASKS

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
        time.sleep(1.0) #REPLACE WITH ACTUAL COMPUTATION

    return 0

if __name__ == '__main__':
    # Specify optional arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--noverbose',dest='verbose',help='reduce output verbosity',action='store_false')
    #arg_parser.add_argument('-i',dest='inputfile')
    args = arg_parser.parse_args()

    # Execute main function
    returnFlag = main(verbose=args.verbose)
    #returnFlag = main(verbose=args.verbose,inputfile=args.inputfile)

    # Exit
    sys.exit(returnFlag)
