import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from attack.attack_utils import mia, poison_attack, check_mia


if __name__ == '__main__':
    INPUT_SIZE = (96, 96, 3)
    FACE_ATTRS = ['Attractive']
    poison_dataset_config = {
        'dataset_name': 'celeba',
        'input_shape': INPUT_SIZE,
        'face_attrs': FACE_ATTRS
    }

    attack_config = {
        'iters': 1000,
        'learning_rate': 0.01,
        'batch_size': 100,
        'if_selection': False
    }

    clean_label_flag = True 
    if clean_label_flag:
        clean_label_str = 'clean'
    else:
        clean_label_str = 'dirty'

    for target_class in range(2**len(FACE_ATTRS)):
        for seed_amount in [5000]:
            for poison_encoder in ['inceptionv3', 'mobilenetv2', 'xception']:
            #for poison_encoder in ['inceptionv3']:
                poison_config = {
                'poison_encoder_name': poison_encoder,
                'poison_img_dir': './poisoning_dataset_{}_label/imgs/'.format(clean_label_str),
                'poison_label_dir': './poisoning_dataset_{}_label/labels/'.format(clean_label_str),
                'anchorpoint_img_dir': './poisoning_dataset_{}_label/anchorpoint_imgs/'.format(clean_label_str),
                'target_class': target_class,
                'seed_amount': seed_amount,
                'anchorpoint_amount': 5000,
                'clean_label_flag': True,
                'fcn_sizes': [128, 2**len(FACE_ATTRS)],
                'transferable_attack_flag': False,
                }
                poison_attack(poison_config,
                              poison_dataset_config,
                              attack_config)