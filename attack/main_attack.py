from __future__ import print_function, division

import os
import gc
import sys
sys.path.append('..')
import numpy as np
import pickle
import tensorflow as tf

EXP_MODEL_ROOT_DIR = './exp_models/'

from utils import check_directory, load_model, save_model, merge_dataset
from attack.attack_utils import save_img_from_array, save_poison_label, load_dataset, load_img_from_dir, summarize_keras_trainable_variables
from attack.dirty_label_attack import dirty_label_attack
from attack.clean_label_attack import clean_label_attack
from build_models import FeatureExtractor, TransferLearningModel
from prepare_data import ExperimentDataset
from tensorflow_privacy import DPKerasAdamOptimizer

DP_SGD_HYPERPARAMETERS = {
    'noise_multiplier':1.,
    'l2_norm_clip': 1.0,
    'num_microbatches': 100,
    'learning_rate': 0.001
}

class PoisonAttack:
    def __init__(self,
                 poison_config,
                 poison_dataset_config,
                 attack_config):
        self.poison_config = poison_config
        self.poison_dataset_config = poison_dataset_config
        self.attack_config = attack_config
        self._prepare_dataset_flag = True
        self._attack_setup()

        self.save_ckpts = False

    def _prepare_dataset(self):
        fe = FeatureExtractor(self.poison_encoder_name,
                              input_shape=self.input_shape)
        self.dataset = ExperimentDataset(dataset_name=self.dataset_name,
                                         preprocess_fn=fe.preprocess_fn,
                                         img_size=self.input_shape,
                                         face_attrs=self.face_attrs)
        self._prepare_dataset_flag = False

    def _update_config(self,
                       poison_config,
                       poison_dataset_config,
                       attack_config):
        self.poison_config = poison_config
        self.poison_config = poison_config
        self.poison_dataset_config = poison_dataset_config
        self.attack_config = attack_config
        self._attack_setup()
        # Update data preprocessor
        if 'vgg16' in self.poison_encoder_name:
            self.dataset.preprocess_fn = tf.keras.applications.vgg16.preprocess_input
            self.dataset.preprocess_type = 'caffe' 
            self.dataset.preprocess_mean = [103.939, 116.779, 123.68]
            self.dataset.image_scale = np.array([np.array([0,255])-x for x in self.dataset.preprocess_mean])\
                                       .transpose()
        elif 'resnet50' in self.poison_encoder_name:
            self.dataset.preprocess_fn = tf.keras.applications.resnet.preprocess_input
            self.dataset.preprocess_type = 'caffe' 
            self.dataset.preprocess_mean = [103.939, 116.779, 123.68]
            self.dataset.image_scale = np.array([np.array([0,255])-x for x in self.dataset.preprocess_mean])\
                                       .transpose()
        elif 'inceptionv3' in self.poison_encoder_name:
            self.dataset.preprocess_fn = tf.keras.applications.inception_v3.preprocess_input
            self.dataset.preprocess_type = 'tensorflow' 
            self.dataset.preprocess_mean = [0, 0, 0]
            self.dataset.image_scale = np.array([[-1., 1.] for i in range(3)])\
                                       .transpose()
        elif 'mobilenetv2' in self.poison_encoder_name:
            self.dataset.preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
            self.dataset.preprocess_type = 'tensorflow' 
            self.dataset.preprocess_mean = [0, 0, 0]
            self.dataset.image_scale = np.array([[-1., 1.] for i in range(3)])\
                                       .transpose()
        elif 'xception' in self.poison_encoder_name:
            self.dataset.preprocess_fn = tf.keras.applications.xception.preprocess_input
            self.dataset.preprocess_type = 'tensorflow' 
            self.dataset.preprocess_mean = [0, 0, 0]
            self.dataset.image_scale = np.array([[-1., 1.] for i in range(3)])\
                                       .transpose()
        
    def _attack_setup(self):

        self.output_img_flag = True

        POISON_CFGS = ['poison_encoder_name',
                       'poison_img_dir',
                       'poison_label_dir',
                       'target_class',
                       'seed_amount',
                       'base_amount',
                       'fcn_sizes',
                       'attack_type']

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
        self.base_amount = self.poison_config['base_amount']
        self.fcn_sizes = self.poison_config['fcn_sizes']
        self.attack_type = self.poison_config['attack_type']

        if self.attack_type == 'clean_label':
            attack_type_str = 'clean_label_attack'
        elif self.attack_type == 'dirty_label':
            attack_type_str = 'dirty_label_attack'
        elif self.attack_type == 'adversarial_examples':
            attack_type_str = 'adversarial_examples'
        elif self.attack_type == 'adversarial_examples_collision':
            attack_type_str = 'adversarial_examples_collision'
        elif 'watermarking' in self.attack_type:
            if 'opacity' in self.attack_config.keys():
                opacity = self.attack_config['opacity']
            else:
                opacity = 0.1
            self.attack_type = 'watermarking_{}'.format(opacity)
            attack_type_str = 'watermarking_{}'.format(opacity)
            
        else:
            raise NotImplementedError("Unknown attack type: {}".format(self.attack_type))

        base_img_root_dir = self.poison_config['base_img_dir']

        if 'output_img_flag' in self.poison_config.keys():
            self.output_img_flag = self.poison_config['output_img_flag']

        self.target_encoder_name = self.poison_config['poison_encoder_name']

        self.dataset_name = self.poison_dataset_config['dataset_name']
        self.input_shape = self.poison_dataset_config['input_shape']
        if 'face_attrs' in self.poison_dataset_config.keys():
            self.face_attrs = self.poison_dataset_config['face_attrs']
            print(self.face_attrs)
        else:
            self.face_attrs = None
        if self._prepare_dataset_flag:
            self._prepare_dataset()

        # Balancing the attack result
        self.poison_amount = self.seed_amount
        self.seed_amount = self.seed_amount - int(self.seed_amount/self.dataset.num_classes)
        self.base_amount = self.seed_amount
        self.base_amount = np.min([self.base_amount,
                                          len(self.dataset.get_attack_dataset(target_class=self.target_class)[1])])
        self.balancing_amount = np.max([self.poison_amount - self.base_amount, 0])

        if self.attack_type == 'dirty_label':
            poison_img_sub_dir = '{}/{}_{}_{}'\
                                .format(self.dataset_name,
                                        self.target_class,
                                        self.seed_amount,
                                        self.base_amount)
            poison_label_sub_dir = '{}/{}_{}_{}'\
                                .format(self.dataset_name,
                                        self.target_class,
                                        self.seed_amount,
                                        self.base_amount)
        else:
            poison_img_sub_dir = '{}/{}/{}_{}_{}'\
                                    .format(self.poison_encoder_name,
                                            self.dataset_name,
                                            self.target_class,
                                            self.seed_amount,
                                            self.base_amount)
            poison_label_sub_dir = '{}/{}/{}_{}_{}'\
                                    .format(self.poison_encoder_name,
                                        self.dataset_name,
                                        self.target_class,
                                        self.seed_amount,
                                        self.base_amount)


        self.poison_img_dir = os.path.join(poison_img_root_dir, poison_img_sub_dir)
        self.base_img_dir = os.path.join(base_img_root_dir, poison_img_sub_dir)

        self.poison_label_dir = os.path.join(poison_label_root_dir, poison_label_sub_dir)
        self.poison_label_path = os.path.join(self.poison_label_dir, 'poison_label.npy')

        self.poison_img_dir = self.poison_img_dir.replace('_finetuned', '')
        self.base_img_dir = self.base_img_dir.replace('_finetuned', '')

        self.poison_label_dir = self.poison_label_dir.replace('_finetuned', '')
        self.poison_label_path = self.poison_label_path.replace('_finetuned', '')

        self.poison_img_dir = self.poison_img_dir.replace('dp_', '')
        self.base_img_dir = self.base_img_dir.replace('dp_', '')

        self.poison_label_dir = self.poison_label_dir.replace('dp_', '')
        self.poison_label_path = self.poison_label_path.replace('dp_', '')

        self.clean_model_dir = os.path.join(EXP_MODEL_ROOT_DIR,
                                             'clean_model/')
        self.clean_model_path = os.path.join(self.clean_model_dir,
                                             '{}_{}.h5'.format(self.dataset_name,
                                                               self.poison_encoder_name))
            

        self.poisoned_model_dir = os.path.join(EXP_MODEL_ROOT_DIR,
                                               attack_type_str,
                                               poison_img_sub_dir)

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
                            self.base_img_dir = self.poison_img_dir.replace('imgs', 'base_imgs')
                            break

            if 'watermarking' in self.attack_type:
                self.poison_img_dir = self.poison_img_dir.replace('watermarking', self.attack_type)
                self.poison_label_path = self.poison_label_path.replace('watermarking', self.attack_type)
                self.base_img_dir = self.base_img_dir.replace('watermarking', self.attack_type)

    def get_base_dataset(self):
        imgs = load_img_from_dir(self.base_img_dir,
                                 self.seed_amount)
        imgs = self.dataset._preprocess_imgs(imgs)
        labels = self.dataset._to_onehot(self.target_class*np.ones((len(imgs),1)))
        return (imgs, labels)

    def get_poison_dataset(self):
        print('Fetching poisoning dataset ({} attack)...'.format(self.attack_type))
        print(self.poison_img_dir)
        balancing_dataset = self.dataset.get_attack_dataset(target_class=self.target_class,
                                                            data_range=[-self.balancing_amount,None])
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
        if self.attack_type == 'adversarial_examples':
            poison_dataset = adversarial_example_attack(clean_model=self.get_clean_model(),
                                                        attack_dataset=attack_dataset,
                                                        target_class=self.target_class,
                                                        poison_amount=self.base_amount)
            if not self.output_img_flag:
                return poison_dataset

            check_directory(self.poison_img_dir)
            save_img_from_array(poison_dataset[0],
                                self.poison_img_dir,
                                preprocess_type=fe.preprocess_type,
                                preprocess_mean=fe.preprocess_mean)

            check_directory(self.poison_label_dir)
            save_poison_label(poison_dataset[1], self.poison_label_path)
            poison_dataset = merge_dataset(poison_dataset, balancing_dataset)

            return poison_dataset
            
        if self.attack_type == 'adversarial_examples_collision':
            def _build_surrogate_model():
                input = tf.keras.Input(shape=self.input_shape)
                x = FeatureExtractor('xception', input_shape=self.input_shape).model(input)
                x = tf.keras.layers.Dense(10, activation='softmax')(x)
                return tf.keras.Model(input, x)

            poison_dataset = adversarial_example_attack(clean_model=self.get_clean_model(),#_build_surrogate_model(),
                                                        attack_dataset=attack_dataset,
                                                        target_class=self.target_class,
                                                        batch_size=50,
                                                        poison_amount=self.base_amount)
            if not self.output_img_flag:
                return poison_dataset

            check_directory(self.poison_img_dir)
            save_img_from_array(poison_dataset[0],
                                self.poison_img_dir,
                                preprocess_type=fe.preprocess_type,
                                preprocess_mean=fe.preprocess_mean)

            check_directory(self.poison_label_dir)
            save_poison_label(poison_dataset[1], self.poison_label_path)
            poison_dataset = merge_dataset(poison_dataset, balancing_dataset)

            return poison_dataset 

        if self.attack_type == 'dirty_label':

            poison_dataset = dirty_label_attack(target_class=self.target_class,
                                                attack_dataset=attack_dataset,
                                                poison_amount=self.base_amount)

            if not self.output_img_flag:
                return poison_dataset

            check_directory(self.poison_img_dir)
            save_img_from_array(poison_dataset[0],
                                self.poison_img_dir,
                                preprocess_type=fe.preprocess_type,
                                preprocess_mean=fe.preprocess_mean)

            check_directory(self.poison_label_dir)
            save_poison_label(poison_dataset[1], self.poison_label_path)
            poison_dataset = merge_dataset(poison_dataset, balancing_dataset)

            return poison_dataset

        if self.attack_type == 'clean_label':

            poison_dataset, base_dataset  = clean_label_attack(encoder=encoder,
                                                               target_class=self.target_class,
                                                               attack_dataset=attack_dataset,
                                                               seed_amount=self.seed_amount,
                                                               base_amount=self.base_amount,
                                                               image_scale=fe.image_scale,
                                                               poison_config=self.attack_config)
        else:
            print("Unknown attack type!")
            exit(0)

        del encoder
        gc.collect()

        if self.output_img_flag:
            check_directory(self.poison_img_dir)
            poisons = poison_dataset[0]
            save_img_from_array(poisons,
                                self.poison_img_dir,
                                preprocess_type=fe.preprocess_type,
                                preprocess_mean=fe.preprocess_mean)
                                
            check_directory(self.base_img_dir)
            bases = base_dataset[0]
            save_img_from_array(bases,
                                self.base_img_dir,
                                preprocess_type=fe.preprocess_type,
                                preprocess_mean=fe.preprocess_mean)

            check_directory(self.poison_label_dir)
            print(self.poison_label_dir)
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
        print("Get clean model...")
        print(self.clean_model_path)
        if os.path.exists(self.clean_model_path):
            return load_model(self.clean_model_path)
        else:
            clean_dataset = self.dataset.get_member_dataset()
            if 'finetuned' in self.target_encoder_name and 'dp' in self.target_encoder_name:
                print('With DP')
                dp_opt = DPKerasAdamOptimizer(
                             **DP_SGD_HYPERPARAMETERS
                         )
                tl = TransferLearningModel(self.target_encoder_name,
                                           self.input_shape,
                                           self.fcn_sizes,
                                           optimizer=dp_opt,
                                           loss_fn=tf.keras.losses.CategoricalCrossentropy(
                                               reduction=tf.losses.Reduction.NONE
                                           ))
            elif 'finetuned' in self.target_encoder_name:
                tl = TransferLearningModel(self.target_encoder_name,
                                           self.input_shape,
                                           self.fcn_sizes,
                                           optimizer=tf.keras.optimizers.Adam(lr=1e-5))
            elif 'dp' in self.target_encoder_name:
                print('With DP')
                dp_opt = DPKerasAdamOptimizer(
                             **DP_SGD_HYPERPARAMETERS
                         )
                
                tl = TransferLearningModel(self.target_encoder_name,
                                           self.input_shape,
                                           self.fcn_sizes,
                                           optimizer=dp_opt,
                                           loss_fn=tf.keras.losses.CategoricalCrossentropy(
                                               reduction=tf.losses.Reduction.NONE
                                           ))
            else:
                tl = TransferLearningModel(self.target_encoder_name,
                                           self.input_shape,
                                           self.fcn_sizes)
            tl.transfer_learning(clean_dataset, 
                                 save_ckpts=self.save_ckpts, 
                                 ckpt_info="{}/clean_model/".format(self.dataset_name)
                                 )
            check_directory(self.clean_model_dir)
            save_model(tl.model, self.clean_model_path)
            with open(self.clean_model_path+'.history.pkl', 'wb') as handle:
                pickle.dump(tl.tl_history, handle)
            return tl.model

    def get_poisoned_model(self):
        print("Get poisoned model...")
        print(self.poisoned_model_path)
        if os.path.exists(self.poisoned_model_path):
            return load_model(self.poisoned_model_path)
        else:
            clean_dataset = self.dataset.get_member_dataset()
            poison_dataset = self.get_poison_dataset()
            training_dataset = merge_dataset(clean_dataset, poison_dataset)

            if 'finetuned' in self.target_encoder_name and 'dp' in self.target_encoder_name:
                print('With DP')
                dp_opt = DPKerasAdamOptimizer(
                             **DP_SGD_HYPERPARAMETERS
                         )
                tl = TransferLearningModel(self.target_encoder_name,
                                           self.input_shape,
                                           self.fcn_sizes,
                                           optimizer=dp_opt,
                                           loss_fn=tf.keras.losses.CategoricalCrossentropy(
                                               reduction=tf.losses.Reduction.NONE
                                           ))
            elif 'finetuned' in self.target_encoder_name:
                tl = TransferLearningModel(self.target_encoder_name,
                                           self.input_shape,
                                           self.fcn_sizes,
                                           optimizer=tf.keras.optimizers.Adam(lr=1e-5))
            elif 'dp' in self.target_encoder_name:
                print('With DP')
                dp_opt = DPKerasAdamOptimizer(
                             **DP_SGD_HYPERPARAMETERS
                         )
                
                tl = TransferLearningModel(self.target_encoder_name,
                                           self.input_shape,
                                           self.fcn_sizes,
                                           optimizer=dp_opt,
                                           loss_fn=tf.keras.losses.CategoricalCrossentropy(
                                               reduction=tf.losses.Reduction.NONE
                                           ))
            else:
                tl = TransferLearningModel(self.target_encoder_name,
                                        self.input_shape,
                                        self.fcn_sizes)
            summarize_keras_trainable_variables(tl.model, ': start poisoning model')
            tl.transfer_learning(training_dataset,save_ckpts=self.save_ckpts,
                                 ckpt_info="{}/{}/target_{}/".format(self.dataset_name,
                                                                     self.attack_type, 
                                                                     self.target_class))
            summarize_keras_trainable_variables(tl.model, ': after poisoning model')
            check_directory(self.poisoned_model_dir)
            with open(self.poisoned_model_path+'.history.pkl', 'wb') as handle:
                pickle.dump(tl.tl_history, handle)
            save_model(tl.model, self.poisoned_model_path)
            return tl.model
