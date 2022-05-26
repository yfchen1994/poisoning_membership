import pickle

########
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dataset', type=str, default='stl10')
parser.add_argument('--encoder', type=str, default='inceptionv3')
parser.add_argument('--device_no', type=str, default='0')
parser.add_argument('--seed_amount', type=int, default=400)
parser.add_argument('--attack_type', type=str, default='clean_label')
parser.add_argument('--check_mia', type=bool, default=False)

args = parser.parse_args()
########

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= args.device_no
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from attack.main_attack import PoisonAttack
from attack.attack_utils import mia, check_mia, poison_attack

import numpy as np
import gc

if __name__ == '__main__':
    INPUT_SIZE = (96, 96, 3)


    poison_dataset_config = {
        'dataset_name': args.dataset,
        'input_shape': INPUT_SIZE,
    }

    if args.dataset == 'celeba':
        poison_dataset_config['face_attrs'] = ['Attractive']
        print(poison_dataset_config)

    if args.dataset in ['stl10', 'mnist', 'cifar10']:
        output_shape = 10
    else:
        output_shape = 2

    attack_config = {
        'iters': 1000,
        'learning_rate': 0.01,
        'batch_size': 200,
    }

    attack_type = args.attack_type 

    poison_config = {
        'poison_encoder_name': args.encoder,
        'poison_img_dir': './poisoning_dataset_{}/imgs/'.format(args.attack_type),
        'poison_label_dir': './poisoning_dataset_{}/labels/'.format(args.attack_type),
        'base_img_dir': './poisoning_dataset_{}/base_imgs/'.format(args.attack_type),
        'target_class': args.target_class,
        'seed_amount': args.seed_amount,
        'base_amount': args.seed_amount,
        'attack_type': args.attack_type,
        'fcn_sizes': [128, output_shape],
    }
    if args.check_mia:
        check_mia(poison_config=poison_config,
                  poison_dataset_config=poison_dataset_config,
                  attack_config=attack_config)
    else:
        poison_attack(poison_config=poison_config,
                    poison_dataset_config=poison_dataset_config,
                    attack_config=attack_config)
