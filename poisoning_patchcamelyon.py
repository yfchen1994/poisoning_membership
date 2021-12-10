import pickle
import gc
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--encoder', type=str, default='inceptionv3')
parser.add_argument('--device_no', type=str, default='0')
parser.add_argument('--seed_amount', type=int, default=5000)

args = parser.parse_args()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= args.device_no
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from attack.attack_utils import mia, check_mia, poison_attack

def calculate_mia():

    INPUT_SIZE = (96, 96, 3)

    poison_dataset_config = {
        'dataset_name': 'patchcamelyon',
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

    for target_class in range(1):
        for seed_amount in [args.seed_amount]:
            for poison_encoder in ['inceptionv3']:
                poison_config = {
                'poison_encoder_name': poison_encoder,
                'poison_img_dir': './poisoning_dataset_{}_label/imgs/'.format(clean_label_str),
                'poison_label_dir': './poisoning_dataset_{}_label/labels/'.format(clean_label_str),
                'anchorpoint_img_dir': './poisoning_dataset_{}_label/anchorpoint_imgs/'.format(clean_label_str),
                'target_class': target_class,
                'seed_amount': seed_amount,
                'anchorpoint_amount': seed_amount,
                'clean_label_flag': clean_label_flag,
                'fcn_sizes': [128, 2],
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
    
    seed_amount = args.seed_amount

    for target_class in [0,1]:
        member_dataset = attack.dataset.get_member_dataset(target_class=target_class)
        nonmember_dataset = attack.dataset.get_nonmember_dataset(target_class=target_class)
        testing_dataset = attack.dataset.get_nonmember_dataset()
        for poison_encoder in ENCODERS:
            poison_config = {
            'poison_encoder_name': poison_encoder,
            'poison_img_dir': './poisoning_dataset_{}_label/imgs/'.format(clean_label_str),
            'poison_label_dir': './poisoning_dataset_{}_label/labels/'.format(clean_label_str),
            'anchorpoint_img_dir': './poisoning_dataset_{}_label/anchorpoint_imgs/'.format(clean_label_str),
            'target_class': target_class,
            'seed_amount': seed_amount,
            'anchorpoint_amount': seed_amount,
            'clean_label_flag': clean_label_flag,
            'fcn_sizes': [128, 2],
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
    with open('patchcamelyon_results_{}.pkl'.format(args.seed_amount), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    calculate_mia()
    exit(0)
    INPUT_SIZE = (96, 96, 3)

    poison_dataset_config = {
        'dataset_name': 'patchcamelyon',
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
        for seed_amount in [args.seed_amount]:
            for encoder in [args.encoder]:
                    poison_config = {
                        'poison_encoder_name': encoder,
                        'poison_img_dir': './poisoning_dataset_{}_label/imgs/'.format(clean_label_str),
                        'poison_label_dir': './poisoning_dataset_{}_label/labels/'.format(clean_label_str),
                        'anchorpoint_img_dir': './poisoning_dataset_{}_label/anchorpoint_imgs/'.format(clean_label_str),
                        'target_class': target_class,
                        'seed_amount': seed_amount,
                        'anchorpoint_amount': seed_amount,
                        'clean_label_flag': clean_label_flag,
                        'fcn_sizes': [128, 2],
                        'transferable_attack_flag': False,
                        'target_encoder_name': encoder,
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