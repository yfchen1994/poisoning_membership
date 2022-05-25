import sys
sys.path.append('..')
sys.path.append('.')

import tensorflow as tf
import numpy as np
import os

import gc
from utils import check_directory, TrainingAccuracyPerClass, TrainingLossPerClass
import PIL

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
                 input_shape):
        self.name = pretrained_model_name
        self.input_shape = input_shape
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
            self.model = tf.keras.applications\
                            .resnet.ResNet50(**PRETRAINED_MODEL_SETTING)
            self.preprocess_fn = tf.keras.applications.resnet.preprocess_input
            self.preprocess_type = 'caffe' 
            self.preprocess_mean = [103.939, 116.779, 123.68]
            self.image_scale = np.array([np.array([0,255])-x for x in self.preprocess_mean])\
                                 .transpose()

        elif 'xception' in pretrained_model_name:
            self.model = tf.keras.applications\
                            .xception.Xception(**PRETRAINED_MODEL_SETTING)
            self.preprocess_fn = tf.keras.applications.xception.preprocess_input
            self.preprocess_type = 'tensorflow' 
            self.preprocess_mean = [0, 0, 0]
            self.image_scale = np.array([[-1., 1.] for i in range(3)])\
                                 .transpose()
        
        elif 'inceptionv3' in pretrained_model_name:
            self.model = tf.keras.applications\
                            .inception_v3.InceptionV3(**PRETRAINED_MODEL_SETTING)
            self.preprocess_fn = tf.keras.applications.inception_v3.preprocess_input
            self.preprocess_type = 'tensorflow' 
            self.preprocess_mean = [0, 0, 0]
            self.image_scale = np.array([[-1., 1.] for i in range(3)])\
                                 .transpose()
        
        elif 'mobilenetv2' in pretrained_model_name:
            self.model = tf.keras.applications\
                            .mobilenet_v2.MobileNetV2(**PRETRAINED_MODEL_SETTING)
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



if __name__ == '__main__':
    pass