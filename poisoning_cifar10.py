import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from attack.main_attack import PoisonAttack
from attack.attack_utils import mia, check_mia, poison_attack

import numpy as np
import gc

ENCODER = 'inceptionv3'

TARGET_CLASS = 3
INPUT_SIZE = (96, 96, 3)

poison_dataset_config = {
    'dataset_name': 'cifar10',
    'input_shape': INPUT_SIZE,
}



def attack_cifar10(poison_config):

    attack_config = {
        'iters': 1000,
        'learning_rate': 0.01,
        'batch_size': 100,
        'if_selection': False
    }

    attack = PoisonAttack(poison_config,
                          poison_dataset_config,
                          attack_config)
    member_dataset = attack.dataset.get_member_dataset(target_class=TARGET_CLASS)
    nonmember_dataset = attack.dataset.get_nonmember_dataset(target_class=TARGET_CLASS)
    print("Working on clean model.")
    model = attack.get_clean_model()
    clean_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr')
    del model
    gc.collect()
    print("Poisoning model.")
    model = attack.get_poisoned_model()
    poison_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr')
    del model
    gc.collect()
    print("Diff:{}".format(poison_auc-clean_auc))


if __name__ == '__main__':
    INPUT_SIZE = (96, 96, 3)
    poison_dataset_config = {
        'dataset_name': 'cifar10',
        'input_shape': INPUT_SIZE,
    }

    attack_config = {
        'iters': 1000,
        'learning_rate': 0.01,
        'batch_size': 100,
        'if_selection': False
    }

    for target_class in range(10):
        for seed_amount in [1000]:
            for poison_encoder in ['inceptionv3', 'mobilenetv2', 'xception']:
            #for poison_encoder in ['inceptionv3']:
                poison_config = {
                'poison_encoder_name': poison_encoder,
                'poison_img_dir': './poisoning_dataset_clean_label/imgs/',
                'poison_label_dir': './poisoning_dataset_clean_label/labels/',
                'anchorpoint_img_dir': './poisoning_dataset_clean_label/anchorpoint_imgs/',
                'target_class': target_class,
                'seed_amount': seed_amount,
                'anchorpoint_amount': 1000,
                'clean_label_flag': True,
                'fcn_sizes': [128, 10],
                'transferable_attack_flag': False,
                }
                poison_attack(poison_config,
                              poison_dataset_config,
                              attack_config)

                """
                check_mia(poison_config,
                        poison_dataset_config,
                        attack_config,
                        target_class)
                """
