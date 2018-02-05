'''
Created on 5 lut 2018

@author: mgdak
'''

import os
import argparse
import sys
from pprint import pprint
import shutil as sh
import random

def main(args):
    pprint(args)
    
    topDirs = os.listdir(args.src_dir)

    # This would print all the files and directories
    for classDir in topDirs:
        print(classDir)
        files = os.listdir(args.src_dir + "\\" + classDir)
        start = int(len(files)*args.split_ratio)
        random.shuffle(files)
        for j in range(start,0,-1):
            os.makedirs(args.dst_dir + "\\" + classDir,
                        exist_ok=True)
            sh.move(args.src_dir + "\\" + classDir + "\\" + files[j],
                    args.dst_dir + "\\" + classDir + "\\" + files[j])

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--split_ratio', type=float,
        help='Defines the split ratio from validation which has to be moved to the test'
        , default=0.5)
        
    parser.add_argument('--dst_dir', type=str,
        help='Destination directory for splited data set', default='./car_ims/tst')
    
    parser.add_argument('--src_dir', type=str,
        help='Source directory where validation data set is located', default='./car_ims/val')
    
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))