import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from attack.attack_utils import mia, check_mia, poison_attack

if __name__ == '__main__':
    #for seed_amount in [40, 80, 200]:
    INPUT_SIZE = (96, 96, 3)

    poison_dataset_config = {
        'dataset_name': 'patchcamelyon',
        'input_shape': INPUT_SIZE,
    }

    attack_config = {
        'iters': 1000,
        'learning_rate': 0.02,
        'batch_size': 100,
        'if_selection': False
    }

    clean_label_flag = True 

    if clean_label_flag:
        clean_label_str = 'clean'
    else:
        clean_label_str = 'dirty'

    for target_class in range(2):
        for seed_amount in [5000]:
            for encoder in ['inceptionv3', 'mobilenetv2', 'xception']:
                    poison_config = {
                        'poison_encoder_name': encoder,
                        'poison_img_dir': './poisoning_dataset_{}_label/imgs/'.format(clean_label_str),
                        'poison_label_dir': './poisoning_dataset_{}_label/labels/'.format(clean_label_str),
                        'anchorpoint_img_dir': './poisoning_dataset_{}_label/anchorpoint_imgs/'.format(clean_label_str),
                        'target_class': target_class,
                        'seed_amount': seed_amount,
                        'anchorpoint_amount': 5000,
                        'clean_label_flag': clean_label_flag,
                        'fcn_sizes': [128, 2],
                        'transferable_attack_flag': False,
                        'target_encoder_name': encoder,
                    }
                    """
                    poison_attack(poison_config=poison_config,
                                    poison_dataset_config=poison_dataset_config,
                                    attack_config=attack_config)
                    """
                    check_mia(poison_config=poison_config,
                                poison_dataset_config=poison_dataset_config,
                                attack_config=attack_config,
                                target_class=target_class)
