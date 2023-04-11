import sys
sys.path.append("../")

from src.sslHiC import get_DCIs
from src.data.utils import cool_to_mats, mtx_to_mat, npz_to_mat, npy_to_mat

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--path1", type=str, required=True, help='Path to first Hi-C data, supposing to be .cool/.mtx/.npy/.npz')
    parser.add_argument("--path2", type=str, required=True, help='Path to second Hi-C data, supposing to be .cool/.mtx/.npy/.npz')
    parser.add_argument("--resol", type=str, required=True, choices=['500kb','50kb','10kb'], help= 'Resolution of Hi-C data, supposing to be 500kb/50kb/10kb')
    parser.add_argument("--chr", nargs='+', type=str, required=False, help='Target chromosome (required if using .cool files)')
    parser.add_argument("--key", type=str, required=False, help='Key of the data (required if using .npz files)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.path1.split('.')[-1] == 'cool' and args.path2.split('.')[-1] == 'cool':
        if args.chr is None:
            raise Exception("Please specify the chr if using cool files")
        mat1 = cool_to_mats(args.path1, args.chr)
        mat2 = cool_to_mats(args.path2, args.chr)
        for i, chr in enumerate(args.chr):
            print(f"The results are saving in {chr}_DCIs.npy")
            props = get_DCIs(mat1[i], mat2[i], args.resol)
            np.save(f'{chr}.npy', props)
    else:
        if args.path1.split('.')[-1] == 'mtx' and args.path2.split('.')[-1] == 'mtx':
            mat1 = mtx_to_mat(args.path1)
            mat2 = mtx_to_mat(args.path2)
            print(f"The results are saving in DCIs.npy")
        elif args.path1.split('.')[-1] == 'npz' and args.path2.split('.')[-1] == 'npz':
            mat1 = npz_to_mat(args.path1, args.key)
            mat2 = npz_to_mat(args.path2, args.key)
            print(f"The results are saving in DCIs.npy")
        elif args.path1.split('.')[-1] == 'npy' and args.path2.split('.')[-1] == 'npy':
            mat1 = npy_to_mat(args.path1)
            mat2 = npy_to_mat(args.path2)
            print(f"The results are saving in DCIs.npy")
        else:
            raise NotImplementedError("Please specify the correct input format")
        props = get_DCIs(mat1, mat2, args.resol)
        np.save(f'DCIs.npy', props)
        
