#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
import os

# Globals
SCATTER = False
PRINT = True

# Functions
def plotterHelp(exitcode):
    print('plotter.py [-s] file')
    print('This command creates a python plot based on file specified')
    print('in the first argument which should be the path to a file')
    print('Options: ')
    print('-s: link the points together, default is scatter')
    print('-h: help; prints this message')
    sys.exit(exitcode)

def parseFile(Path):
    datafile = open(Path, 'r')

    evo = []
    for line in datafile:
        line = line[:-1]
        if line[0:2] == 'n:':
            words = line.split(' ')
            evo.append([words[2],words[6],words[10],words[14]])
    if PRINT:
        print(evo)
    datafile.close()
    data = 1
    return data
    

# Parse Command-line Options
args = sys.argv[1:]
while len(args) and args[0].startswith('-') and len(args[0]) > 1:
    arg = args.pop(0)
    if arg == '-h':
        plotterHelp(0)
    elif arg == '-s':
        SCATTER = True
    else:
        sys.stderr.write('Error: Invalid argument\n')
        plotterHelp(1)

if len(args) >=1:
    Path = args.pop(0)
else:
    sys.stderr.write('Error: No file path given\n')
    plotterHelp(1)

data = parseFile(Path)
