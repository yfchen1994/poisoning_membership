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
from attack.main_attack import PoisonAttack
from attack.attack_utils import mia, check_mia, poison_attack

import numpy as np
import gc

def attack_stl10(poison_config):

    attack_config = {
        'iters': 1000,
        'learning_rate': 0.01,
        'batch_size': 100,
        'if_selection': False
    }

    attack = PoisonAttack(poison_config,
                          poison_dataset_config,
                          attack_config)
    """
    member_dataset = attack.dataset.get_member_dataset(target_class=TARGET_CLASS)
    nonmember_dataset = attack.dataset.get_nonmember_dataset(target_class=TARGET_CLASS)
    print("Working on clean model.")
    model = attack.get_clean_model()
    clean_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr')
    del model
    gc.collect()
    """
    print("Poisoning model.")
    model = attack.get_poisoned_model()
    #poison_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr')
    del model
    gc.collect()
    #print("Diff:{}".format(poison_auc-clean_auc))

def calculate_mia():

    INPUT_SIZE = (96, 96, 3)
    for target_class in range(6,10):

        poison_dataset_config = {
            'dataset_name': 'stl10',
            'input_shape': INPUT_SIZE,
        }

        attack_config = {
            'iters': 1000,
            'learning_rate': 0.02,
            'batch_size': 150,
            'if_selection': False
        }

    seed_amount = 400

    for target_class in range(1):
        for poison_encoder in ['inceptionv3']:
            poison_config = {
            'poison_encoder_name': poison_encoder,
            'poison_img_dir': './poisoning_dataset_clean_label/imgs/',
            'poison_label_dir': './poisoning_dataset_clean_label/labels/',
            'anchorpoint_img_dir': './poisoning_dataset_clean_label/anchorpoint_imgs/',
            'target_class': target_class,
            'seed_amount': seed_amount,
            'anchorpoint_amount': 400,
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
            'anchorpoint_amount': 400,
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
    with open('stl10_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    #for seed_amount in [40, 80, 200]:
    calculate_mia()
    exit(0)
    INPUT_SIZE = (96, 96, 3)

    poison_dataset_config = {
        'dataset_name': 'stl10',
        'input_shape': INPUT_SIZE,
    }

    attack_config = {
        'iters': 1000,
        'learning_rate': 0.02,
        'batch_size': 150,
        'if_selection': False
    }

    clean_label_flag = True

    if clean_label_flag:
        clean_label_str = 'clean'
    else:
        clean_label_str = 'dirty'

    for target_class in [args.target_class]:
        for seed_amount in [400]:
            #for encoder in ['inceptionv3', 'mobilenetv2', 'xception']:
            for encoder in [args.encoder]:
                poison_config = {
                    'poison_encoder_name': encoder,
                    'poison_img_dir': './poisoning_dataset_{}_label/imgs/'.format(clean_label_str),
                    'poison_label_dir': './poisoning_dataset_{}_label/labels/'.format(clean_label_str),
                    'anchorpoint_img_dir': './poisoning_dataset_{}_label/anchorpoint_imgs/'.format(clean_label_str),
                    'target_class': target_class,
                    'seed_amount': seed_amount,
                    'anchorpoint_amount': 400,
                    'clean_label_flag': clean_label_flag,
                    'fcn_sizes': [128, 10],
                    'transferable_attack_flag': False,
                }
                poison_attack(poison_config=poison_config,
                              poison_dataset_config=poison_dataset_config,
                              attack_config=attack_config)
                """
                check_mia(poison_config=poison_config,
                          poison_dataset_config=poison_dataset_config,
                          attack_config=attack_config,
                          target_class=target_class)
                """
                        #attack_stl10(poison_config=poison_config)