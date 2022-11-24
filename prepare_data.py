import sys
sys.path.append('..')
sys.path.append('.')

import tensorflow as tf
import numpy as np
import os
import pandas as pd
from attack.attack_utils import load_img_from_dir, save_img_from_array, load_poison_label
import gc
import PIL

# Please change the dataset folder if needed.
DATASET_ROOT = './dataset'

class ExperimentDataset:
    def __init__(self,
                 dataset_name,
                 preprocess_fn,
                 img_size,
                 face_attrs=None):
        self.dataset_name = dataset_name
        self.preprocess_fn = preprocess_fn
        self.img_size = img_size
        self.face_attrs = face_attrs
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        if self.dataset_name == 'cifar10':
            self.member_amount = 10000
            self._prepare_cifar10()
        elif self.dataset_name == 'cifar10_stl10':
            self.member_amount = 10000
            self._prepare_cifar10_stl10()
        elif self.dataset_name == 'celeba':
            self.member_amount = 10000
            self._prepare_celeba()
        elif self.dataset_name == 'mnist':
            self.member_amount = 10000
            self._prepare_mnist()
        elif self.dataset_name == 'stl10':
            self.member_amount = 4000
            self._prepare_stl10()
        elif self.dataset_name == 'patchcamelyon':
            self.member_amount = 10000
            self._prepare_patchcamelyon()
        else:
            raise NotImplementedError("Unsupported dataset ({})"\
                                       .format(self.dataset_name))
    
    def _get_balanced_data(self, train_x, train_y, start_range, end_range):
        x = None
        y = None
        for i in range(self.num_classes):
            sub_x = train_x[np.where(train_y==i)]
            sub_x = sub_x[start_range:end_range]
            if self.dataset_name == 'mnist':
                sub_x = self._grayscale_to_rgb(sub_x)

            sub_y = train_y[np.where(train_y==i)]
            sub_y = sub_y[start_range:end_range]

            if x is None:
                x = sub_x
                y = sub_y
            else:
                x = np.r_[x, sub_x]
                y = np.r_[y, sub_y]
        return (x,y)

    def _prepare_cifar10(self):
        self.num_classes = 10
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
        train_y = train_y.reshape((-1))
        test_y = test_y.reshape((-1))
        sub_class_amount = int(self.member_amount/self.num_classes)
        self._member_dataset_ori = self._get_balanced_data(train_x,
                                                           train_y,
                                                           0,
                                                           sub_class_amount)
        self._nonmember_dataset_ori = self._get_balanced_data(train_x,
                                                              train_y,
                                                              sub_class_amount,
                                                              2*sub_class_amount)
        self._attack_dataset_ori = (test_x, test_y)

    def _prepare_cifar10_stl10(self):
        self.num_classes = 10
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
        train_y = train_y.reshape((-1))
        test_y = test_y.reshape((-1))
        sub_class_amount = int(self.member_amount/self.num_classes)
        self._member_dataset_ori = self._get_balanced_data(train_x,
                                                           train_y,
                                                           0,
                                                           sub_class_amount)
        self._nonmember_dataset_ori = self._get_balanced_data(train_x,
                                                              train_y,
                                                              sub_class_amount,
                                                              2*sub_class_amount)

        self.num_classes = 10
        stl10_dir = os.path.join(DATASET_ROOT, 'stl10_binary')
        train_x_path = os.path.join(stl10_dir, 'train_X.bin')
        train_y_path = os.path.join(stl10_dir, 'train_y.bin')
        test_x_path = os.path.join(stl10_dir, 'test_X.bin')
        test_y_path = os.path.join(stl10_dir, 'test_y.bin')

        def _read_labels(label_path):
            with open(label_path, 'rb') as f:
                return (np.fromfile(f, dtype=np.uint8)-1)

        def _read_imgs(img_path):
            with open(img_path, 'rb') as f:
                x = np.fromfile(f, dtype=np.uint8)
                x = np.reshape(x, (-1, 3, 96, 96))
                x = np.transpose(x, (0, 3, 2, 1))
                return x

        train_x = _read_imgs(train_x_path)
        train_y = _read_labels(train_y_path)

        self._attack_dataset_ori = (train_x, train_y)

        del train_x, train_y
        gc.collect()

        test_x = _read_imgs(test_x_path)
        test_y = _read_labels(test_y_path)

        amount_to_complement = sub_class_amount - np.sum(self._attack_dataset_ori[1]==0)
        _complement_dataset = self._get_balanced_data(test_x,
                                                      test_y,
                                                      0,
                                                      amount_to_complement)
        self._attack_dataset_ori = (np.r_[self._attack_dataset_ori[0],
                                          _complement_dataset[0]],
                                    np.r_[self._attack_dataset_ori[1],
                                          _complement_dataset[1]])
        del test_x, test_y, _complement_dataset
        gc.collect()

    def _prepare_mnist(self):
        self.num_classes = 10
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

        sub_class_amount = int(self.member_amount/self.num_classes)
        self._member_dataset_ori = self._get_balanced_data(train_x,
                                                           train_y,
                                                           0,
                                                           sub_class_amount)

        self._nonmember_dataset_ori = self._get_balanced_data(train_x,
                                                              train_y,
                                                              sub_class_amount,
                                                              2*sub_class_amount)

        self._attack_dataset_ori = (self._grayscale_to_rgb(test_x), test_y)

    def _prepare_patchcamelyon(self):
        self.num_classes = 2
        x_path = os.path.join(DATASET_ROOT, 'patchcamelyon/camelyonpatch_level_2_split_test_x.h5')
        y_path = os.path.join(DATASET_ROOT, 'patchcamelyon/camelyonpatch_level_2_split_test_y.h5')
        import h5py
        train_x = h5py.File(x_path, 'r').get('x')[:]
        train_y = h5py.File(y_path, 'r').get('y')[:]
        train_y = train_y.reshape(-1)

        sub_class_amount = int(self.member_amount/self.num_classes)
        self._member_dataset_ori = self._get_balanced_data(train_x,
                                                           train_y,
                                                           0,
                                                           sub_class_amount)
        self._nonmember_dataset_ori = self._get_balanced_data(train_x,
                                                              train_y,
                                                              sub_class_amount,
                                                              2*sub_class_amount)
        self._attack_dataset_ori = self._get_balanced_data(train_x,
                                                           train_y,
                                                           2*sub_class_amount,
                                                           3*sub_class_amount)
    
    def _prepare_celeba(self):
        self.num_classes = 2**len(self.face_attrs)
        img_dir = os.path.join(DATASET_ROOT, 'celeba/img_align_celeba/')
        attrs_file_path = os.path.join(DATASET_ROOT, 'celeba/list_attr_celeba.csv')
        attrs_complete = pd.read_csv(attrs_file_path)
        attrs_selected = attrs_complete.loc[:5*self.member_amount, :]
        labels = np.zeros(len(attrs_selected))
        for face_attr in self.face_attrs:
            labels = 2*labels + (attrs_selected.loc[:,face_attr].to_numpy()+1)/2
        labels = labels.astype('int8')

        def _read_celeba_imgs(paths):
            imgs = []
            for path in paths:
                img_path = os.path.join(img_dir, path)
                img = np.array(PIL.Image.open(img_path))
                imgs.append(img)
            return np.array(imgs)

        sub_class_amount = int(self.member_amount/self.num_classes)
        self._member_dataset_ori = self._get_balanced_data(attrs_selected.iloc[:,0].to_numpy(),
                                                           labels,
                                                           0,
                                                           sub_class_amount)
        self._member_dataset_ori = (_read_celeba_imgs(self._member_dataset_ori[0]),
                                    self._member_dataset_ori[1])
        self._nonmember_dataset_ori = self._get_balanced_data(attrs_selected.iloc[:,0].to_numpy(),
                                                               labels,
                                                               sub_class_amount,
                                                               2*sub_class_amount)

        self._nonmember_dataset_ori = (_read_celeba_imgs(self._nonmember_dataset_ori[0]),
                                        self._nonmember_dataset_ori[1])

        self._attack_dataset_ori = self._get_balanced_data(attrs_selected.iloc[:,0].to_numpy(),
                                                           labels,
                                                           2*sub_class_amount,
                                                           3*sub_class_amount)
        self._attack_dataset_ori = (_read_celeba_imgs(self._attack_dataset_ori[0]),
                                    self._attack_dataset_ori[1])
        
    def _prepare_stl10(self):
        self.num_classes = 10
        stl10_dir = os.path.join(DATASET_ROOT, 'stl10_binary')
        train_x_path = os.path.join(stl10_dir, 'train_X.bin')
        train_y_path = os.path.join(stl10_dir, 'train_y.bin')
        test_x_path = os.path.join(stl10_dir, 'test_X.bin')
        test_y_path = os.path.join(stl10_dir, 'test_y.bin')

        def _read_labels(label_path):
            with open(label_path, 'rb') as f:
                return (np.fromfile(f, dtype=np.uint8)-1)

        def _read_imgs(img_path):
            with open(img_path, 'rb') as f:
                x = np.fromfile(f, dtype=np.uint8)
                x = np.reshape(x, (-1, 3, 96, 96))
                x = np.transpose(x, (0, 3, 2, 1))
                return x

        train_x = _read_imgs(train_x_path)
        train_y = _read_labels(train_y_path)

        sub_class_amount = int(self.member_amount/self.num_classes)
        self._member_dataset_ori = self._get_balanced_data(train_x,
                                                           train_y,
                                                           0,
                                                           sub_class_amount)
        self._attack_dataset_ori = self._get_balanced_data(train_x,
                                                           train_y,
                                                           sub_class_amount,
                                                           None) 

        del train_x, train_y
        gc.collect()

        test_x = _read_imgs(test_x_path)
        test_y = _read_labels(test_y_path)
        self._nonmember_dataset_ori = self._get_balanced_data(test_x,
                                                              test_y,
                                                              -sub_class_amount,
                                                              None)

        amount_to_complement = sub_class_amount - np.sum(self._attack_dataset_ori[1]==0)
        _complement_dataset = self._get_balanced_data(test_x,
                                                      test_y,
                                                      0,
                                                      amount_to_complement)
        self._attack_dataset_ori = (np.r_[self._attack_dataset_ori[0],
                                          _complement_dataset[0]],
                                    np.r_[self._attack_dataset_ori[1],
                                          _complement_dataset[1]])
        del test_x, test_y, _complement_dataset
        gc.collect()

    def get_member_dataset(self, 
                           target_class=None,
                           data_range=None,
                           return_idx=False):
        return self._process_dataset((self._member_dataset_ori[0],
                                      self._member_dataset_ori[1]),
                                      target_class=target_class,
                                      data_range=data_range,
                                      return_idx=return_idx)
    
    def get_nonmember_dataset(self,
                              target_class=None,
                              data_range=None,
                              return_idx=False):
        return self._process_dataset((self._nonmember_dataset_ori[0],
                                      self._nonmember_dataset_ori[1]),
                                     target_class=target_class,
                                     data_range=data_range,
                                     return_idx=return_idx)
    
    def get_attack_dataset(self,
                           target_class=None,
                           data_range=None,
                           return_idx=False):
        return self._process_dataset((self._attack_dataset_ori[0],
                                      self._attack_dataset_ori[1]),
                                      target_class=target_class,
                                      data_range=data_range,
                                      return_idx=return_idx)
    
    def _process_dataset(self, 
                         dataset, 
                         target_class=None,
                         data_range=None,
                         return_idx=False):
        if target_class is None:
            pass
        else:
            select_idx = np.where(dataset[1].reshape(-1)==target_class)
            dataset = (dataset[0][select_idx],
                       dataset[1][select_idx])

        if not (type(data_range) in [list, tuple]):
            data_range = (0, data_range)

        if return_idx:
            return (self._preprocess_imgs(dataset[0][data_range[0]:data_range[1]]),
                    self._to_onehot(dataset[1][data_range[0]:data_range[1]]),
                    select_idx[data_range[0]:data_range[1]])
        else:
            return (self._preprocess_imgs(dataset[0][data_range[0]:data_range[1]]),
                    self._to_onehot(dataset[1][data_range[0]:data_range[1]]))
    
    def _resize_imgs(self, x):
        x = tf.constant(x)
        x = tf.image.resize(x, 
                            self.img_size[:2], 
                            method='bicubic')
        return tf.saturate_cast(x, dtype=tf.uint8).numpy()

    def _preprocess_imgs(self, x):
        return self.preprocess_fn(self._resize_imgs(x))

    def _grayscale_to_rgb(self, x):
        x = tf.constant(x.reshape((*x.shape,1)))
        x = tf.image.grayscale_to_rgb(x)
        return tf.saturate_cast(x, dtype=tf.uint8).numpy()
        
    def _to_onehot(self, y):
        return tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
