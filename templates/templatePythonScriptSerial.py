#!/usr/bin/env python

'''
Template for serial python script

Commands to run script:
$ sourcePYTHON3

$ ./templatePythonScriptSerial.py
or 
$ python templatePythonScriptSerial.py
'''

__author__ = "Dries Allaerts"
__date__   = "October 8, 2018"

import numpy as np
import os
import sys
import time
import pandas as pd
import argparse

def main(verbose):

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
