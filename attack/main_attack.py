from __future__ import print_function, division

import os
import gc
import sys
sys.path.append('..')
import numpy as np
import pickle

EXP_MODEL_ROOT_DIR = './exp_models/'

from utils import check_directory, load_model, save_model, merge_dataset
from attack.attack_utils import save_img_from_array, save_poison_label, load_dataset, load_img_from_dir
from attack.dirty_label_attack import dirty_label_attack
from attack.clean_label_attack import clean_label_attack
from models.build_models import FeatureExtractor, ExperimentDataset, TransferLearningModel

class PoisonAttack:
    def __init__(self,
                 poison_config,
                 poison_dataset_config,
                 attack_config):
        self.poison_config = poison_config
        self.poison_dataset_config = poison_dataset_config
        self.attack_config = attack_config
        self._attack_setup()

    def _prepare_dataset(self):
        fe = FeatureExtractor(self.poison_encoder_name, 
                              input_shape=self.input_shape)
        self.dataset = ExperimentDataset(dataset_name=self.dataset_name,
                                         preprocess_fn=fe.preprocess_fn,
                                         img_size=self.input_shape,
                                         face_attrs=self.face_attrs) 

    def _attack_setup(self):
        
        self.clean_label_flag = True
        self.output_img_flag = True

        POISON_CFGS = ['poison_encoder_name',
                       'poison_img_dir',
                       'poison_label_dir',
                       'target_class',
                       'seed_amount',
                       'anchorpoint_amount',
                       'fcn_sizes']

        POISONING_DATASET_CFGS = ['input_shape',
                                'dataset_name']

        for key in POISON_CFGS:
            if key not in self.poison_config.keys():
                raise ValueError("Key {} is not included for \
                                the poison configurations".format(key))

        for key in POISONING_DATASET_CFGS:
            if key not in self.poison_dataset_config.keys():
                raise ValueError("Key {} is not included for \
                                the dataset configurations".format(key))

        self.poison_encoder_name = self.poison_config['poison_encoder_name']
        poison_img_root_dir = self.poison_config['poison_img_dir']
        poison_label_root_dir = self.poison_config['poison_label_dir']
        self.target_class = self.poison_config['target_class']
        self.seed_amount = self.poison_config['seed_amount']
        self.anchorpoint_amount = self.poison_config['anchorpoint_amount']
        self.fcn_sizes = self.poison_config['fcn_sizes']

        anchorpoint_img_root_dir = self.poison_config['anchorpoint_img_dir']
        
        if 'clean_label_flag' in self.poison_config.keys():
            self.clean_label_flag = self.poison_config['clean_label_flag']

        if 'output_img_flag' in self.poison_config.keys():
            self.output_img_flag = self.poison_config['output_img_flag']
        if 'transferable_attack_flag' in self.poison_config.keys():
            self.transferable_attack_flag = self.poison_config['transferable_attack_flag']
        else:
            self.transferable_attack_flag = False

        if self.transferable_attack_flag:
            if self.poison_config['target_encoder_name'] == self.poison_encoder_name:
                self.transferable_attack_flag = False
                self.target_encoder_name = self.poison_config['poison_encoder_name']
            else:
                self.target_encoder_name = self.poison_config['target_encoder_name']
                print("="*10)
                ("Transferable attack...")
                print("Target encoder: {}".format(self.target_encoder_name))
                print("Poisoning encoder: {}".format(self.poison_encoder_name))
                print("="*10)
        else:
            self.target_encoder_name = self.poison_config['poison_encoder_name']

        self.dataset_name = self.poison_dataset_config['dataset_name']
        self.input_shape = self.poison_dataset_config['input_shape']
        if 'face_attrs' in self.poison_dataset_config.keys():
            self.face_attrs = self.poison_dataset_config['face_attrs']
            print(self.face_attrs)
        else:
            self.face_attrs = None
        
        self._prepare_dataset()

        # Balancing the attack result
        self.poison_amount = self.seed_amount
        self.seed_amount = self.seed_amount - int(self.seed_amount/self.dataset.num_classes)
        self.anchorpoint_amount = self.seed_amount
        self.anchorpoint_amount = np.min([self.anchorpoint_amount, 
                                          len(self.dataset.get_attack_dataset(target_class=self.target_class)[1])])
        self.balancing_amount = np.max([self.poison_amount - self.anchorpoint_amount, 0])

        if self.clean_label_flag:
            poison_img_sub_dir = '{}/{}/{}_{}_{}'\
                                .format(self.poison_encoder_name,
                                        self.dataset_name,
                                        self.target_class,
                                        self.seed_amount,
                                        self.anchorpoint_amount)
            poison_label_sub_dir = '{}/{}/{}_{}_{}'\
                               .format(self.poison_encoder_name,
                                       self.dataset_name,
                                       self.target_class,
                                       self.seed_amount,
                                       self.anchorpoint_amount) 
        else:
            poison_img_sub_dir = '{}/{}_{}_{}'\
                                .format(self.dataset_name,
                                        self.target_class,
                                        self.seed_amount,
                                        self.anchorpoint_amount)
            poison_label_sub_dir = '{}/{}_{}_{}'\
                                .format(self.dataset_name,
                                        self.target_class,
                                        self.seed_amount,
                                        self.anchorpoint_amount) 

        self.poison_img_dir = os.path.join(poison_img_root_dir, poison_img_sub_dir)
        self.anchorpoint_img_dir = os.path.join(anchorpoint_img_root_dir, poison_img_sub_dir)

        self.poison_label_dir = os.path.join(poison_label_root_dir, poison_label_sub_dir)
        self.poison_label_path = os.path.join(self.poison_label_dir, 'poison_label.npy')

        self.clean_model_dir = os.path.join(EXP_MODEL_ROOT_DIR,
                                             'clean_model/')
        self.clean_model_path = os.path.join(self.clean_model_dir,
                                             '{}_{}.h5'.format(self.dataset_name,
                                                               self.poison_encoder_name))

        if self.clean_label_flag:
            clean_label_flag_str = 'clean_label_attack'        
        else:
            clean_label_flag_str = 'dirty_label_attack'        

        self.poisoned_model_dir = os.path.join(EXP_MODEL_ROOT_DIR,
                                               clean_label_flag_str,
                                               poison_img_sub_dir)

        if self.transferable_attack_flag:
            self.poisoned_model_path = os.path.join(self.poisoned_model_dir,
                                                    '{}_{}_{}_poisoned_model.h5'.format(self.dataset_name,
                                                                                        self.target_encoder_name,
                                                                                        self.poison_encoder_name))
        else:
            self.poisoned_model_path = os.path.join(self.poisoned_model_dir,
                                                    '{}_{}_poisoned_model.h5'.format(self.dataset_name,
                                                                                    self.poison_encoder_name))

        if self.output_img_flag:
            # Find whether there exist a poisoning dataset whose poison amount
            # satisfies the required.
            dir_to_check = '/'.join(self.poison_img_dir.split('/')[:-1])
            if os.path.exists(dir_to_check):
                subdirs = os.listdir(dir_to_check)
                for subdir in subdirs:
                    if subdir.startswith(str(self.target_class)):
                        supported_poison_amount = int(subdir.split('_')[1])
                        if self.seed_amount <= supported_poison_amount:
                            self.poison_img_dir = os.path.join(dir_to_check, subdir)
                            # Change the poisoning label path.
                            self.poison_label_path = os.path.join(self.poison_img_dir.replace('imgs', 'labels'),
                                                                  'poison_label.npy')
                            self.anchorpoint_img_dir = self.poison_img_dir.replace('imgs', 'anchorpoint_imgs')
                            break
    
    def get_anchorpoint_dataset(self):
        print(self.anchorpoint_img_dir)
        imgs = load_img_from_dir(self.anchorpoint_img_dir,
                                 self.seed_amount)
        imgs = self.dataset._preprocess_imgs(imgs)
        labels = self.dataset._to_onehot(self.target_class*np.ones((len(imgs),1)))
        return (imgs, labels)

    def get_poison_dataset(self):
        balancing_dataset = self.dataset.get_attack_dataset(target_class=self.target_class,
                                                            data_range=[-self.balancing_amount,None])
        print('balancing dataset...')
        print(balancing_dataset[0].shape)
        if self.output_img_flag:
            # Load queries from images
            if os.path.exists(self.poison_img_dir):
                poison_dataset = load_dataset(self.poison_img_dir,
                                                self.poison_label_path, 
                                                self.seed_amount)
                poison_dataset = (self.dataset._preprocess_imgs(poison_dataset[0]),poison_dataset[1])
                poison_dataset = merge_dataset(poison_dataset, balancing_dataset)
                return poison_dataset

        # Load the attack model
        fe = FeatureExtractor(self.poison_encoder_name, 
                              input_shape=self.input_shape)
        encoder = fe.model
        encoder.trainable = False
        attack_dataset = ExperimentDataset(dataset_name=self.dataset_name,
                                           preprocess_fn=fe.preprocess_fn,
                                           img_size=self.input_shape,
                                           face_attrs=self.face_attrs).get_attack_dataset()

        # Debugging
        #check_GPU_memory("After load the attack_model...")
        if not self.clean_label_flag:
    
            poison_dataset = dirty_label_attack(target_class=self.target_class,
                                                attack_dataset=attack_dataset,
                                                poison_amount=self.anchorpoint_amount)
            
            if not self.output_img_flag:
                return poison_dataset

            check_directory(self.poison_img_dir)
            save_img_from_array(poison_dataset[0],
                                   self.poison_img_dir) 

            check_directory(self.poison_label_dir)
            save_poison_label(poison_dataset[1], self.poison_label_path)
            poison_dataset = merge_dataset(poison_dataset, balancing_dataset)

            return poison_dataset

        poison_dataset, anchorpoint_dataset, anchorpoint_idx = clean_label_attack(encoder=encoder,
                                                                                  target_class=self.target_class,
                                                                                  attack_dataset=attack_dataset,
                                                                                  seed_amount=self.seed_amount,
                                                                                  anchorpoint_amount=self.anchorpoint_amount,
                                                                                  poison_config=self.attack_config)

        del encoder 
        gc.collect()

        if self.output_img_flag:
            check_directory(self.poison_img_dir)
            poisons = poison_dataset[0]
            save_img_from_array(poisons,
                                self.poison_img_dir)

            check_directory(self.anchorpoint_img_dir)
            anchorpoints = anchorpoint_dataset[0]
            save_img_from_array(anchorpoints,
                                self.anchorpoint_img_dir)
            np.save(os.path.join(self.anchorpoint_img_dir, 'idx.npy'),
                    anchorpoint_idx)

            check_directory(self.poison_label_dir)
            poison_label = poison_dataset[1]
            save_poison_label(poison_label, self.poison_label_path)

            poison_dataset = load_dataset(self.poison_img_dir, 
                                          self.poison_label_path, 
                                          img_num=self.seed_amount)
            poison_dataset = (self.dataset._preprocess_imgs(poison_dataset[0]),
                              poison_dataset[1])
            poison_dataset = merge_dataset(poison_dataset, balancing_dataset)

            return poison_dataset

        return poison_dataset 

    def get_clean_model(self):
        if os.path.exists(self.clean_model_path):
            return load_model(self.clean_model_path)
        else:
            clean_dataset = self.dataset.get_member_dataset()
            tl = TransferLearningModel(self.target_encoder_name,
                                       self.input_shape,
                                       self.fcn_sizes)
            tl.transfer_learning(clean_dataset)
            check_directory(self.clean_model_dir)
            save_model(tl.model, self.clean_model_path)
            with open(self.clean_model_path+'.history.pkl', 'wb') as handle:
                pickle.dump(tl.tl_history, handle)
            return tl.model

    def get_poisoned_model(self):
        if os.path.exists(self.poisoned_model_path):
            return load_model(self.poisoned_model_path)
        else:
            clean_dataset = self.dataset.get_member_dataset()
            poison_dataset = self.get_poison_dataset()
            training_dataset = merge_dataset(clean_dataset, poison_dataset)

            tl = TransferLearningModel(self.target_encoder_name,
                                       self.input_shape,
                                       self.fcn_sizes)
            tl.transfer_learning(training_dataset)
            check_directory(self.poisoned_model_dir)
            print(tl.tl_history)
            with open(self.poisoned_model_path+'.history.pkl', 'wb') as handle:
                pickle.dump(tl.tl_history, handle)
            save_model(tl.model, self.poisoned_model_path)
            return tl.model