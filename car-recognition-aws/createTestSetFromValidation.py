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
        files = os.listdir(args.src_dir + "/" + classDir)
        train_start = int(len(files)*args.train_split_ratio)
        tst_start = int(len(files)*args.tst_split_ratio)
        random.shuffle(files)
        #move train data
        for j in range(train_start + tst_start, tst_start, -1):
            os.makedirs(args.train_dst_dir + "/" + classDir, exist_ok=True)
            sh.move(args.src_dir + "/" + classDir + "/" + files[j],
                    args.train_dst_dir + "/" + classDir + "/" + files[j])

        #move tst data
        for j in range(tst_start, 0, -1):
            os.makedirs(args.tst_dst_dir + "/" + classDir, exist_ok=True)
            sh.move(args.src_dir + "/" + classDir + "/" + files[j],
                    args.tst_dst_dir + "/" + classDir + "/" + files[j])

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--tst_split_ratio', type=float,
        help='Defines the split ratio from validation which has to be moved to the test'
        , default=0.25)

    parser.add_argument('--train_split_ratio', type=float,
        help='Defines the split ratio from validation which has to be moved to the train'
        , default=0.5)
        
    parser.add_argument('--tst_dst_dir', type=str,
        help='Destination directory for splited test data set', default='./car_ims/tst')
    
    parser.add_argument('--train_dst_dir', type=str,
        help='Destination directory for splited train data set', default='./car_ims/train')
        
    parser.add_argument('--src_dir', type=str,
        help='Source directory where validation data set is located', default='./car_ims/val')
    
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))