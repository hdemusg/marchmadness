import pandas as pd
import sklearn
import numpy as np
import sys
import argparse

from brackets import generate_bracket

parser = argparse.ArgumentParser()

def main():
    parser.add_argument("-d", "--datafile", help="Training Data Filename")
    parser.add_argument("-o", "--outputfile", help="Output Filename")
    args = parser.parse_args()
    if args.datafile == None:
        d = "mm_train.xlsx"
    else:
        d = args.datafile
    if args.outputfile == None:
        o = "mm_preds.xlsx"
    else:
        o = args.outputfile
    generate_bracket(d, o)

if __name__ == "__main__":
    main()