import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--encoder', type=str, default='inceptionv3')
parser.add_argument('--device_no', type=str, default='0')
parser.add_argument('--seed_amount', type=int, default=1000)
parser.add_argument('--attack_type', type=str, default='clean_label')
parser.add_argument('--check_mia', type=bool, default=False)

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

def calculate_mia():

    INPUT_SIZE = (96, 96, 3)
    poison_dataset_config = {
        'dataset_name': 'cifar10',
        'input_shape': INPUT_SIZE,
    }

    attack_config = {
        'iters': 1000,
        'learning_rate': 0.02,
        'batch_size': 100,
        'if_selection': False
    }

    attack_type = args.attack_type

    for target_class in range(1):
        for seed_amount in [1000]:
            for poison_encoder in ['inceptionv3']:
                poison_config = {
                'poison_encoder_name': poison_encoder,
                'poison_img_dir': './poisoning_dataset_{}/imgs/'.format(attack_type),
                'poison_label_dir': './poisoning_dataset_{}/labels/'.format(attack_type),
                'anchorpoint_img_dir': './poisoning_dataset_{}/anchorpoint_imgs/'.format(attack_type),
                'target_class': target_class,
                'seed_amount': seed_amount,
                'anchorpoint_amount': 1000,
                'attack_type': attack_type, 
                'fcn_sizes': [128, 10],
                'transferable_attack_flag': False,
                }
    
    clean_model_auc = {}
    dirty_model_auc = {}
    clean_model_acc = {}
    dirty_model_acc = {}
    ENCODERS = ['vgg16','resnet50', 'inceptionv3', 'mobilenetv2', 'xception']
    #ENCODERS = [encoder + '_finetuned' for encoder in ENCODERS]
    #ENCODERS = ['dp_'+encoder for encoder in ENCODERS]
    for encoder in ENCODERS:
        clean_model_auc[encoder] = []
        dirty_model_auc[encoder] = []
        dirty_model_acc[encoder] = []

    from attack.main_attack import PoisonAttack
    from attack.attack_utils import mia, evaluate_model
    attack = PoisonAttack(poison_config,
                          poison_dataset_config,
                          attack_config)

    seed_amount = args.seed_amount
    print(attack_type)
    for target_class in range(10):
        for poison_encoder in ENCODERS:
            poison_config = {
            'poison_encoder_name': poison_encoder,
            'poison_img_dir': './poisoning_dataset_{}/imgs/'.format(attack_type),
            'poison_label_dir': './poisoning_dataset_{}/labels/'.format(attack_type),
            'anchorpoint_img_dir': './poisoning_dataset_{}/anchorpoint_imgs/'.format(attack_type),
            'target_class': target_class,
            'seed_amount': seed_amount,
            'anchorpoint_amount': seed_amount,
            'attack_type': attack_type,
            'fcn_sizes': [128, 10],
            'transferable_attack_flag': False,
            }
            attack._update_config(poison_config=poison_config,
                                  poison_dataset_config=poison_dataset_config,
                                  attack_config=attack_config)
            member_dataset = attack.dataset.get_member_dataset(target_class=target_class)
            nonmember_dataset = attack.dataset.get_nonmember_dataset(target_class=target_class)
            testing_dataset = attack.dataset.get_nonmember_dataset()
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
    with open('cifar10_{}_results_{}.pkl'.format(attack_type, args.seed_amount), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    calculate_mia()
    exit(0)
    INPUT_SIZE = (96, 96, 3)
    poison_dataset_config = {
        'dataset_name': 'cifar10',
        'input_shape': INPUT_SIZE,
    }

    attack_config = {
        'iters': 1000,
        'learning_rate': 0.02,
        'batch_size': 150,
        'if_selection': False,
    }

    attack_type = args.attack_type

    if attack_type == 'watermarking':
        attack_config['opacity'] = 0.8

    for target_class in [args.target_class]:
        for seed_amount in [args.seed_amount]:
            #for poison_encoder in ['inceptionv3', 'mobilenetv2', 'xception']:
            for poison_encoder in [args.encoder]:
                poison_config = {
                'poison_encoder_name': poison_encoder,
                'poison_img_dir': './poisoning_dataset_{}/imgs/'.format(attack_type),
                'poison_label_dir': './poisoning_dataset_{}/labels/'.format(attack_type),
                'anchorpoint_img_dir': './poisoning_dataset_{}/anchorpoint_imgs/'.format(attack_type),
                'target_class': target_class,
                'seed_amount': seed_amount,
                'anchorpoint_amount': 1000,
                'attack_type': attack_type,
                'fcn_sizes': [128, 10],
                'transferable_attack_flag': False,
                }

                if args.check_mia:
                    check_mia(poison_config,
                            poison_dataset_config,
                            attack_config,
                            target_class)
                else:
                    poison_attack(poison_config,
                                poison_dataset_config,
                                attack_config)