import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--encoder', type=str, default='inception')
parser.add_argument('--device_no', type=str, default='0')

args = parser.parse_args()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= args.device_no
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"



import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

def calculate_mia():

    INPUT_SIZE = (96, 96, 3)
    for target_class in range(6,10):

        poison_dataset_config = {
            'dataset_name': 'mnist',
            'input_shape': INPUT_SIZE,
        }

        attack_config = {
            'iters': 1000,
            'learning_rate': 0.02,
            'batch_size': 150,
            'if_selection': False
        }

    for target_class in range(1):
        for seed_amount in [1000]:
            for poison_encoder in ['inceptionv3']:
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
    
    clean_model_auc = {}
    dirty_model_auc = {}
    clean_model_acc = {}
    dirty_model_acc = {}
    ENCODERS = ['inceptionv3', 'mobilenetv2', 'xception']
    for encoder in ENCODERS:
        clean_model_auc[encoder] = []
        dirty_model_auc[encoder] = []
        dirty_model_acc[encoder] = []

    from attack.main_attack import PoisonAttack
    from attack.attack_utils import mia, evaluate_model
    attack = PoisonAttack(poison_config,
                          poison_dataset_config,
                          attack_config)


    for target_class in range(10):
        member_dataset = attack.dataset.get_member_dataset(target_class=target_class)
        nonmember_dataset = attack.dataset.get_nonmember_dataset(target_class=target_class)
        testing_dataset = attack.dataset.get_nonmember_dataset()

        for poison_encoder in ENCODERS:
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
            attack._update_config(poison_config=poison_config,
                                  poison_dataset_config=poison_dataset_config,
                                  attack_config=attack_config)
            model = attack.get_clean_model()

            if target_class == 0:
                clean_test_acc = evaluate_model(model, testing_dataset)
                print("Testing accuracy of the whole dataset: {:.2f}% ({})"\
                    .format(clean_test_acc*100, len(testing_dataset[0])))
                clean_model_acc[poison_encoder] = clean_test_acc
            clean_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr')
            clean_model_auc[poison_encoder].append(clean_auc)
            del model
            gc.collect()
            
            model = attack.get_poisoned_model()
            poison_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr')
            poison_test_acc = evaluate_model(model, testing_dataset)
            print("Testing accuracy of the whole dataset: {:.2f}% ({})"\
                .format(poison_test_acc*100, len(testing_dataset[0])))
            dirty_model_acc[poison_encoder].append(poison_test_acc)
            dirty_model_auc[poison_encoder].append(poison_auc)
    results = {
        'clean_model_auc': clean_model_auc,
        'clean_model_acc': clean_model_acc,
        'dirty_model_auc': dirty_model_auc,
        'dirty_model_acc': dirty_model_acc
    }
    with open('mnist_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    calculate_mia()
    exit(0)
    #for seed_amount in [100, 200, 500]:
    #transferable_test()
    INPUT_SIZE = (96, 96, 3)

    clean_label_flag = True

    if clean_label_flag:
        clean_label_str = 'clean'
    else:
        clean_label_str = 'dirty'

    for target_class in [args.target_class]:

        poison_dataset_config = {
            'dataset_name': 'mnist',
            'input_shape': INPUT_SIZE,
        }

        attack_config = {
            'iters': 1000,
            'learning_rate': 0.02,
            'batch_size': 150,
            'if_selection': False
        }

        for seed_amount in [1000]:
            for poison_encoder in [args.encoder]:
            #for poison_encoder in ['inceptionv3']:
                poison_config = {
                'poison_encoder_name': poison_encoder,
                'poison_img_dir': './poisoning_dataset_{}_label/imgs/'.format(clean_label_str),
                'poison_label_dir': './poisoning_dataset_{}_label/labels/'.format(clean_label_str),
                'anchorpoint_img_dir': './poisoning_dataset_{}_label/anchorpoint_imgs/'.format(clean_label_str),
                'target_class': target_class,
                'seed_amount': seed_amount,
                'anchorpoint_amount': 1000,
                'clean_label_flag': clean_label_flag,
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
