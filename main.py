import json
import argparse
from trainer import train
import torch

def main():
    args = setup_parser().parse_args()
    

    with open(args.config) as data_file:
        param = json.load(data_file)
    
    args = vars(args)
    args.update(param)

    seed = args.get("seed", [1993])
    if not isinstance(seed, list):
        seed = [seed]
    args['seed'] = seed


    
    train(args)

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
