import sys
sys.path.append('..')
sys.path.append('.')

from attack.clean_label_attack import clean_label_attack
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from attack.dirty_label_attack import dirty_label_attack
from attack.attack_utils import load_img_from_dir, save_img_from_array, load_poison_label
import gc
from utils import merge_dataset, check_directory, TrainingAccuracyPerClass, TrainingLossPerClass
import PIL
from models.dropout_features import add_dropout_layer

class TransferLearningModel:
    def __init__(self,
                 pretrained_model_name,
                 input_shape,
                 fcn_sizes,
                 if_compile=True,
                 optimizer=tf.keras.optimizers.Adam(),
                 loss_fn=tf.keras.losses.CategoricalCrossentropy()):
        self.input_shape = input_shape
        self.fcn_sizes = fcn_sizes
        self.feature_extractor = FeatureExtractor(pretrained_model_name,
                                                  input_shape)
        self.optimizer=optimizer
        self.loss_fn = loss_fn
        self._if_compile=if_compile
        self._build_transfer_learning_model()
    
    def _build_transfer_learning_model(self):
        tf.random.set_seed(54321)
        inputs = tf.keras.Input(shape=self.input_shape)
        x = self.feature_extractor.model(inputs, training=False)

        for fcn_size in self.fcn_sizes[:-1]:
            x = tf.keras.layers.Dense(fcn_size, activation='tanh')(x)
        outputs = tf.keras.layers.Dense(self.fcn_sizes[-1], activation='softmax')(x)
        self.model = tf.keras.Model(inputs, outputs)
        if self._if_compile:
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                metrics=['accuracy', TrainingAccuracyPerClass(), TrainingLossPerClass()]
            )

    def get_transfer_learning_model(self):
        return self.model

    def transfer_learning(self,
                          train_ds,
                          epochs=20,
                          batch_size=100,
                          save_ckpts=False,
                          ckpt_info=''):
        tf.random.set_seed(12345)
        if save_ckpts:
            # Save the checkpoints
            print(ckpt_info)
            print(self.feature_extractor.name)
            checkpoint_dir = os.path.join("./checkpoints/", ckpt_info, self.feature_extractor.name)
            checkpoint_path = checkpoint_dir+"/cp-{epoch:d}.ckpt"
            check_directory(checkpoint_dir)

            self.model.save_weights(checkpoint_path.format(epoch=0))

            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                verbose=1,
                save_weights_only=True,
                save_freq='epoch',
            )
            self.tl_history = self.model.fit(train_ds[0], train_ds[1], 
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            callbacks=[cp_callback]).history
        else:
            self.tl_history = self.model.fit(train_ds[0], train_ds[1], 
                                            epochs=epochs,
                                            batch_size=batch_size).history

class FeatureExtractor:
    def __init__(self,
                 pretrained_model_name,
                 input_shape,
                 dropout=False,
                 dropout_rate=False):
        self.name = pretrained_model_name
        self.input_shape = input_shape
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self._prepare_pretrained_model()

    def _prepare_pretrained_model(self):

        def _remove_last_two_layers(base_model):
            return tf.keras.models.Model(base_model.input,
                                         base_model.layers[-2].output,
                                         name='feature_extractor')

        if 'finetuned' in self.name:
            if_fine_tune = True
        else:
            if_fine_tune = False

        pretrained_model_name = self.name
        input_shape = self.input_shape

        PRETRAINED_MODEL_SETTING = {
            'input_shape': input_shape,
            'include_top': False,
            'weights': 'imagenet'
        }

        if 'vgg16' in pretrained_model_name:
            self.model = tf.keras.applications\
                         .vgg16.VGG16(**PRETRAINED_MODEL_SETTING)
            self.preprocess_fn = tf.keras.applications.vgg16.preprocess_input
            self.preprocess_type = 'caffe' 
            self.preprocess_mean = [103.939, 116.779, 123.68]
            self.image_scale = np.array([np.array([0,255])-x for x in self.preprocess_mean])\
                                 .transpose()
            
        elif 'resnet50' in pretrained_model_name:
            if self.dropout:
                self.model = add_dropout_layer(tf.keras.applications
                                               .resnet.ResNet50(**PRETRAINED_MODEL_SETTING),
                                               dropout_rate=self.dropout_rate,
                                               keyword='.*act.*')
            else:
                self.model = tf.keras.applications\
                            .resnet.ResNet50(**PRETRAINED_MODEL_SETTING)
            self.preprocess_fn = tf.keras.applications.resnet.preprocess_input
            self.preprocess_type = 'caffe' 
            self.preprocess_mean = [103.939, 116.779, 123.68]
            self.image_scale = np.array([np.array([0,255])-x for x in self.preprocess_mean])\
                                 .transpose()

        elif 'xception' in pretrained_model_name:
            if self.dropout:
                self.model = add_dropout_layer(tf.keras.applications
                                               .xception.Xception(**PRETRAINED_MODEL_SETTING),
                                               dropout_rate=self.dropout_rate,
                                               keyword='.*act.*')
            else:
                self.model = tf.keras.applications\
                            .xception.Xception(**PRETRAINED_MODEL_SETTING)
            self.preprocess_fn = tf.keras.applications.xception.preprocess_input
            self.preprocess_type = 'tensorflow' 
            self.preprocess_mean = [0, 0, 0]
            self.image_scale = np.array([[-1., 1.] for i in range(3)])\
                                 .transpose()
        
        elif 'inceptionv3' in pretrained_model_name:
            if self.dropout:
                self.model = add_dropout_layer(tf.keras.applications
                                               .inception_v3.InceptionV3(**PRETRAINED_MODEL_SETTING),
                                               dropout_rate=self.dropout_rate,
                                               keyword='.*mixed.*')
            else:
                self.model = tf.keras.applications\
                            .inception_v3.InceptionV3(**PRETRAINED_MODEL_SETTING)
            #self.model.add(tf.keras.layers.Flatten())
            self.preprocess_fn = tf.keras.applications.inception_v3.preprocess_input
            self.preprocess_type = 'tensorflow' 
            self.preprocess_mean = [0, 0, 0]
            self.image_scale = np.array([[-1., 1.] for i in range(3)])\
                                 .transpose()
        
        elif 'mobilenetv2' in pretrained_model_name:
            if self.dropout:
                self.model = add_dropout_layer(tf.keras.applications
                                               .mobilenet_v2.MobileNetV2(**PRETRAINED_MODEL_SETTING),
                                               dropout_rate=self.dropout_rate,
                                               keyword='.*project_BN.*')

            else:
                self.model = tf.keras.applications\
                            .mobilenet_v2.MobileNetV2(**PRETRAINED_MODEL_SETTING)
            #self.model.add(tf.keras.layers.GlobalAveragePooling2D())
            #self.model.add(tf.keras.layers.Flatten())
            self.preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
            self.preprocess_type = 'tensorflow' 
            self.preprocess_mean = [0, 0, 0]
            self.image_scale = np.array([[-1., 1.] for i in range(3)])\
                                 .transpose()

        else:
            raise NotImplementedError("Pretraiend model {} is not available."
                                      .format(pretrained_model_name))

        inputs = tf.keras.Input(shape=self.input_shape)
        x = self.model(inputs, training=False)
        for model_name in ['xception', 'mobilenetv2', 'resnet50', 'inceptionv3', 'vgg16']:
            if model_name in  pretrained_model_name:
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                break
        outputs = tf.keras.layers.Flatten(name='flatten')(x) 
        self.model = tf.keras.Model(inputs, outputs, name='feature_extractor')
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy())

        self.model.trainable = if_fine_tune
        #self.model.summary()

DATASET_ROOT = './datasets/'

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
        #self._attack_dataset_ori = (test_x, test_y)

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
                           data_range=None):
        return self._process_dataset((self._member_dataset_ori[0],
                                      self._member_dataset_ori[1]),
                                      target_class=target_class,
                                      data_range=data_range)
    
    def get_nonmember_dataset(self,
                              target_class=None,
                              data_range=None):
        return self._process_dataset((self._nonmember_dataset_ori[0],
                                      self._nonmember_dataset_ori[1]),
                                     target_class=target_class,
                                     data_range=data_range)
    
    def get_attack_dataset(self,
                           target_class=None,
                           data_range=None):
        return self._process_dataset((self._attack_dataset_ori[0],
                                      self._attack_dataset_ori[1]),
                                      target_class=target_class,
                                      data_range=data_range)
    
    def _process_dataset(self, 
                         dataset, 
                         target_class=None,
                         data_range=None):
        if target_class is None:
            pass
        else:
            select_idx = np.where(dataset[1].reshape(-1)==target_class)
            dataset = (dataset[0][select_idx],
                       dataset[1][select_idx])

        if not (type(data_range) in [list, tuple]):
            data_range = (0, data_range)

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

    #test_transfer_learing()
    pretrained_model_names = [#'xception',
                              'inceptionv3',]
                              #'mobilenetv2']
    input_shape = (96,96,3)
    for model_name in pretrained_model_names:
        pretrained = FeatureExtractor(pretrained_model_name=model_name,input_shape=input_shape, dropout=True,
        dropout_rate=0.3).model.summary()