import sys

import tensorflow as tf
from attack.main_attack import PoisonAttack
from attack.attack_utils import mia

import os
import numpy as np
import gc

ENCODER = 'xception'

TARGET_CLASS = 1 
INPUT_SIZE = (96, 96, 3)
FACE_ATTRS = ['Attractive']

poison_dataset_config = {
    'dataset_name': 'celeba',
    'input_shape': INPUT_SIZE,
    'face_attrs': FACE_ATTRS
}

POISON_CONFIG = {
    'encoder_name': ENCODER,
    'poison_img_dir': './poisoning_dataset_dirty_label/imgs/',
    'poison_label_dir': './poisoning_dataset_dirty_label/labels/',
    'anchorpoint_img_dir': './poisoning_dataset_dirty_label/anchorpoint_imgs/',
    'target_class': TARGET_CLASS,
    'seed_amount': 5000,
    'anchorpoint_amount': 5000,
    'clean_label_flag': False,
    'fcn_sizes': [128, 2**len(FACE_ATTRS)]
}

def attack_celeba(poison_config=POISON_CONFIG):

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
    attack_celeba()
