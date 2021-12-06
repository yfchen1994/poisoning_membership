from attack.clean_label_attack import clean_label_attack
import tensorflow as tf
import numpy as np
import os
import sys
import pandas as pd
sys.path.append('..')
from attack.dirty_label_attack import dirty_label_attack
from attack.attack_utils import load_img_from_dir, save_img_from_array,  load_poison_label
import gc
from utils import merge_dataset, check_directory
import PIL

class TransferLearningModel:
    def __init__(self,
                 pretrained_model_name,
                 input_shape,
                 fcn_sizes,
                 loss_fn=tf.keras.losses.CategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam()):
        self.input_shape = input_shape
        self.fcn_sizes = fcn_sizes
        self.feature_extractor = FeatureExtractor(pretrained_model_name,
                                                  input_shape)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self._build_transfer_learning_model()
    
    def _build_transfer_learning_model(self):
        tf.random.set_seed(12345)
        inputs = tf.keras.Input(shape=self.input_shape)
        x = self.feature_extractor.model(inputs, training=False)
        """
        if self.base_model.name in ['xception', 'mobilenetv2']:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten(name='flatten')(x)
        """
        for fcn_size in self.fcn_sizes[:-1]:
            x = tf.keras.layers.Dense(fcn_size, activation='tanh')(x)
        outputs = tf.keras.layers.Dense(self.fcn_sizes[-1], activation='softmax')(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.summary()
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['accuracy']
        )

    def transfer_learning(self,
                          train_ds,
                          epochs=20,
                          batch_size=100):
        tf.random.set_seed(88888)
        self.tl_history = self.model.fit(train_ds[0], train_ds[1], 
                                         epochs=epochs,
                                         batch_size=batch_size).history

class FeatureExtractor:
    def __init__(self,
                 pretrained_model_name,
                 input_shape):
        self.name = pretrained_model_name
        self.input_shape = input_shape
        self._prepare_pretrained_model()

    def _prepare_pretrained_model(self):
        pretrained_model_name = self.name
        input_shape = self.input_shape

        PRETRAINED_MODEL_SETTING = {
            'input_shape': input_shape,
            'include_top': False,
            'weights': 'imagenet'
        }

        if pretrained_model_name == 'vgg16':
            raise NotImplementedError("As the input preprocessing is a bit \
                                       complex for the VGG family, the ResNet is not supported.")

            self.model = tf.keras.applications\
                         .vgg16.VGG16(**PRETRAINED_MODEL_SETTING)
            self.preprocess_fn = tf.keras.applications.vgg16.preprocess_input
            
        elif pretrained_model_name == 'resnet50':
            raise NotImplementedError("As the input preprocessing is a bit \
                                       complex for the ResNet family, the ResNet is not supported.")
            self.model = tf.keras.applications\
                         .resnet.ResNet50(**PRETRAINED_MODEL_SETTING)
            self.preprocess_fn = tf.keras.applications.resnet.preprocess_input

        elif pretrained_model_name == 'xception':
            self.model = tf.keras.applications\
                         .xception.Xception(**PRETRAINED_MODEL_SETTING)
            #self.model.add(tf.keras.layers.GlobalAveragePooling2D())
            #self.model.add(tf.keras.layers.Flatten())
            self.preprocess_fn = tf.keras.applications.xception.preprocess_input
            self.image_scale = [-1., 1.]
        
        elif pretrained_model_name == 'inceptionv3':
            self.model = tf.keras.applications\
                         .inception_v3.InceptionV3(**PRETRAINED_MODEL_SETTING)
            #self.model.add(tf.keras.layers.Flatten())
            self.preprocess_fn = tf.keras.applications.inception_v3.preprocess_input
            self.image_scale = [-1., 1.]
        
        elif pretrained_model_name == 'mobilenetv2':
            self.model = tf.keras.applications\
                         .mobilenet_v2.MobileNetV2(**PRETRAINED_MODEL_SETTING)
            #self.model.add(tf.keras.layers.GlobalAveragePooling2D())
            #self.model.add(tf.keras.layers.Flatten())
            self.preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
            self.image_scale = [-1., 1.]

        else:
            raise NotImplementedError("Pretraiend model {} is not available."
                                      .format(pretrained_model_name))

        inputs = tf.keras.Input(shape=self.input_shape)
        x = self.model(inputs, training=False)
        if pretrained_model_name in ['xception', 'mobilenetv2']:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Flatten(name='flatten')(x) 
        self.model = tf.keras.Model(inputs, outputs, name='feature_extractor')
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy())

        self.model.trainable = False

DATASET_ROOT = '/home/pretrain_inference/datasets/'

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

    def _prepare_cifar10(self):
        self.num_classes = 10
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
        self._member_dataset_ori = (train_x[:self.member_amount], train_y[:self.member_amount])
        self._nonmember_dataset_ori =(train_x[self.member_amount:2*self.member_amount],
                                      train_y[self.member_amount:2*self.member_amount]) 
        self._attack_dataset_ori = (test_x, test_y)

    def _prepare_mnist(self):
        self.num_classes = 10
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
        self._member_dataset_ori = (self._grayscale_to_rgb(train_x[:self.member_amount]),
                                    train_y[:self.member_amount])
        self._nonmember_dataset_ori = (self._grayscale_to_rgb(train_x[self.member_amount:2*self.member_amount]),
                                       train_y[self.member_amount:2*self.member_amount])
        self._attack_dataset_ori = (self._grayscale_to_rgb(test_x), test_y)

    def _prepare_patchcamelyon(self):
        self.num_classes = 2
        x_path = os.path.join(DATASET_ROOT, 'patchcamelyon/camelyonpatch_level_2_split_test_x.h5')
        y_path = os.path.join(DATASET_ROOT, 'patchcamelyon/camelyonpatch_level_2_split_test_y.h5')
        import h5py
        x = h5py.File(x_path, 'r').get('x')
        y = h5py.File(y_path, 'r').get('y')

        self._member_dataset_ori = (x[:self.member_amount],
                                    y[:self.member_amount].reshape((self.member_amount,-1)))
        self._nonmember_dataset_ori = (x[self.member_amount:2*self.member_amount],
                                       y[self.member_amount:2*self.member_amount].reshape((self.member_amount,-1)))
        self._attack_dataset_ori = (x[2*self.member_amount:3*self.member_amount],
                                    y[2*self.member_amount:3*self.member_amount].reshape((self.member_amount,-1)))
    
    def _prepare_celeba(self):
        self.num_classes = 2**len(self.face_attrs)
        img_dir = os.path.join(DATASET_ROOT, 'celeba/img_align_celeba/')
        attrs_file_path = os.path.join(DATASET_ROOT, 'celeba/list_attr_celeba.csv')
        attrs_complete = pd.read_csv(attrs_file_path)
        attrs_selected = attrs_complete.loc[:3*self.member_amount, :]
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

        self._member_dataset_ori = (_read_celeba_imgs(attrs_selected.iloc[:self.member_amount,0]), 
                                     labels[:self.member_amount])
        self._nonmember_dataset_ori = (_read_celeba_imgs(attrs_selected.iloc[self.member_amount:2*self.member_amount,0]), 
                                        labels[self.member_amount:2*self.member_amount])
        self._attack_dataset_ori = (_read_celeba_imgs(attrs_selected.iloc[2*self.member_amount:3*self.member_amount,0]), 
                                    labels[2*self.member_amount:3*self.member_amount])
    
    def _prepare_stl10(self):
        self.num_classes = 10
        stl10_dir = os.path.join(DATASET_ROOT, 'stl10_binary')
        train_x_path = os.path.join(stl10_dir, 'train_X.bin')
        train_y_path = os.path.join(stl10_dir, 'train_y.bin')
        test_x_path = os.path.join(stl10_dir, 'test_X.bin')
        test_y_path = os.path.join(stl10_dir, 'test_y.bin')

        def _read_labels(label_path):
            with open(label_path, 'rb') as f:
                return (np.fromfile(f, dtype=np.uint8)-1).reshape((-1,1))

        def _read_imgs(img_path):
            with open(img_path, 'rb') as f:
                x = np.fromfile(f, dtype=np.uint8)
                x = np.reshape(x, (-1, 3, 96, 96))
                x = np.transpose(x, (0, 3, 2, 1))
                return x

        train_x = _read_imgs(train_x_path)
        train_y = _read_labels(train_y_path)
        self._member_dataset_ori = (train_x[:self.member_amount],
                                    train_y[:self.member_amount])
        self._attack_dataset_ori = (train_x[self.member_amount:],
                                    train_y[self.member_amount:])
        del train_x, train_y
        gc.collect()

        test_x = _read_imgs(test_x_path)
        test_y = _read_labels(test_y_path)
        self._nonmember_dataset_ori = (test_x[-self.member_amount:],
                                       test_y[-self.member_amount:])
        new_attack_data_needed = self.member_amount - len(self._attack_dataset_ori[0])
        self._attack_dataset_ori = (np.r_[self._attack_dataset_ori[0],
                                          test_x[:new_attack_data_needed]],
                                    np.r_[self._attack_dataset_ori[1],
                                          test_y[:new_attack_data_needed]])
        del test_x, test_y
        gc.collect()

    def get_member_dataset(self, 
                           target_class=None,
                           data_amount=None):
            return self._process_dataset((self._member_dataset_ori[0][:data_amount],
                                          self._member_dataset_ori[1][:data_amount]),
                                         target_class=target_class)
    
    def get_nonmember_dataset(self,
                              target_class=None,
                              data_amount=None):
        return self._process_dataset((self._nonmember_dataset_ori[0][:data_amount],
                                      self._nonmember_dataset_ori[1][:data_amount]),
                                     target_class=target_class)
    
    def get_attack_dataset(self,
                           target_class=None,
                           data_amount=None):
        return self._process_dataset((self._attack_dataset_ori[0][:data_amount],
                                      self._attack_dataset_ori[1][:data_amount]),
                                     target_class=target_class)
    
    def _process_dataset(self, 
                         dataset, 
                         target_class=None):
        if target_class is None:
            pass
        else:
            select_idx = np.where(dataset[1].reshape(-1)==target_class)
            dataset = (dataset[0][select_idx],
                       dataset[1][select_idx])
        return (self._preprocess_imgs(dataset[0]),
                self._to_onehot(dataset[1]))
    
    def _resize_imgs(self, x):
        x = tf.constant(x)
        x = tf.image.resize(x, 
                            self.img_size[:2], 
                            method='bicubic')
        return tf.saturate_cast(x, dtype=tf.uint8).numpy()

    def _preprocess_imgs(self, x):
        #return (self._resize_imgs(x) / 255.) * 2. - 1.
        return self.preprocess_fn(self._resize_imgs(x))

    def _grayscale_to_rgb(self, x):
        x = tf.constant(x.reshape((*x.shape,1)))
        x = tf.image.grayscale_to_rgb(x)
        return tf.saturate_cast(x, dtype=tf.uint8).numpy()
        
    def _to_onehot(self, y):
        return tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

def test_transfer_learing():
    input_shape = (96,96,3)
    tl = TransferLearningModel('xception',
                                input_shape,
                                [128,10])
    exp_dataset = ExperimentDataset('cifar10',
                                    tl.base_model.preprocess_fn,
                                    img_size=input_shape)
    member_dataset = exp_dataset.get_member_dataset()
    gc.collect()
    #tl.transfer_learning(member_dataset)

    attack_dataset = exp_dataset.get_attack_dataset()
    poison_dataset = dirty_label_attack(attack_dataset=attack_dataset,
                                        target_class=0)

    test_path = './poisoning_dataset/dirty_label/cifar10/'
    check_directory(test_path)
    save_img_from_array(poison_dataset[0], test_path)
    load_img_from_dir(test_path, 500)
    training_dataset = merge_dataset(poison_dataset, member_dataset)
    tl.transfer_learning(training_dataset)

if __name__ == '__main__':

    test_transfer_learing()
    pretrained_model_names = ['xception',
                              'inceptionv3',
                              'mobilenetv2']
    input_shape = (96,96,3)
    for model_name in pretrained_model_names:
        pretrained = FeatureExtractor(pretrained_model_name=model_name,input_shape=input_shape).model.summary()