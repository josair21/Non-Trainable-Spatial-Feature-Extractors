"""Train either standalone deterministic model on one supported dataset."""

import argparse
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from training import run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=("minirocketbased","visualprimitives"),required=True)
    parser.add_argument("--dataset",choices=("stl10","eurosat","imagenette"),required=True)
    parser.add_argument("--download",action="store_true")
    args=parser.parse_args()
    run(args.model,args.dataset,args.download)


if __name__=="__main__":
    main()
