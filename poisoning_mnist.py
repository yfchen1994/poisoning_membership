import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'



import tensorflow as tf
from attack.main_attack import PoisonAttack
from attack.attack_utils import mia, evaluate_model, check_mia, visualize_features, poison_attack

import numpy as np
import gc


TARGET_CLASS = 3
INPUT_SIZE = (96, 96, 3)

poison_dataset_config = {
    'dataset_name': 'mnist',
    'input_shape': INPUT_SIZE,
}


def attack_mnist(poison_config):

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
    testing_dataset = attack.dataset.get_nonmember_dataset()
    print('='*30)
    print("Working on clean model.")
    model = attack.get_clean_model()
    clean_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr')
    clean_test_acc_sub = evaluate_model(model, nonmember_dataset)
    clean_test_acc = evaluate_model(model, testing_dataset)
    print("Testing accuracy of the target class {:.2f}%".format(clean_test_acc_sub*100))
    print("Testing accuracy of the whole dataset {:.2f}%".format(clean_test_acc*100))
    del model
    gc.collect()
    print("*"*20)
    print("Poisoning model.")
    model = attack.get_poisoned_model()
    poison_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr')
    poisoned_test_acc_sub = evaluate_model(model, nonmember_dataset)
    poisoned_test_acc = evaluate_model(model, testing_dataset)
    print("Testing accuracy of the target class {:.2f}%".format(poisoned_test_acc_sub*100))
    print("Testing accuracy of the whole dataset {:.2f}%".format(poisoned_test_acc*100))

    del model
    gc.collect()
    print("Diff:{}".format(poison_auc-clean_auc))
    print('='*30)

def transferable_test():
    for seed_amount in [1000]:
        for ENCODER in ['inceptionv3', 'mobilenetv2', 'xception']:
            for target_encoder in ['inceptionv3', 'mobilenetv2', 'xception']:
                if ENCODER == target_encoder:
                    continue
                else:
                    POISON_CONFIG = {
                    'poison_encoder_name': ENCODER,
                    'poison_img_dir': './poisoning_dataset_clean_label/imgs/',
                    'poison_label_dir': './poisoning_dataset_clean_label/labels/',
                    'anchorpoint_img_dir': './poisoning_dataset_clean_label/anchorpoint_imgs/',
                    'target_class': TARGET_CLASS,
                    'seed_amount': seed_amount,
                    'anchorpoint_amount': 1000,
                    'clean_label_flag': True,
                    'fcn_sizes': [128, 10],
                    'transferable_attack_flag': True,
                    'target_encoder_name': target_encoder,
                    }
                    attack_mnist(poison_config=POISON_CONFIG)

if __name__ == '__main__':
    #for seed_amount in [100, 200, 500]:
    #transferable_test()
    INPUT_SIZE = (96, 96, 3)
    for target_class in range(10):

        poison_dataset_config = {
            'dataset_name': 'mnist',
            'input_shape': INPUT_SIZE,
        }

        attack_config = {
            'iters': 1000,
            'learning_rate': 0.02,
            'batch_size': 100,
            'if_selection': False
        }

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
