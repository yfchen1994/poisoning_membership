import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
PROJ_DIR = '../../'

import numpy as np
import tensorflow as tf
TF_DTYPE = tf.float32
if TF_DTYPE is tf.float16:
    print("Setting float16 precision")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.keras.backend.set_floatx('float16')
import gc
from recolor import *
sys.path.append(PROJ_DIR)
from models.build_models import *
from attack.attack_utils import mia, evaluate_model, save_img_from_array
from losses import *
from tensorflow.keras import datasets, layers, models
import timeit

from PIL import Image

class MetaAttack:
    def __init__(self,
                 surrogate_models,
                 meta_shadow_dataset,
                 meta_attack_dataset,
                 target_class,
                 surrogate_model_pattern='duplicate',
                 surrogate_amount=None,
                 model_compile_args=None,
                 training_args={
                     'batch_size': 100,
                 },
                 attack_steps=60,
                 meta_epochs=20,
                 unroll_steps=2,
                 pert_eps=0.08,
                 color_gridshape=[16,32,32],
                 colorpert_eps=0.04,
                 adv_learning_rate=0.01,
                 colorperturb_smooth_lambda=0.05,
                 verbose=True,
                 fine_tune=False,
                 ):
        
        self.verbose = verbose      
        self.fine_tune = fine_tune

        # ----------------------------------------------
        # |       Meta Learning configurations         |
        # ----------------------------------------------
        self.model_compile_args = model_compile_args
        self.surrogate_amount = surrogate_amount
        if surrogate_model_pattern == 'duplicate':
            self.surrogate_models = [
                tf.keras.models.clone_model(surrogate_models)  # Clone_model will reinitilize the weights
                for _ in range(surrogate_amount)
                ]
            self.model_skeleton = tf.keras.models.clone_model(surrogate_models)
            for model in self.surrogate_models:
                model.compile(**model_compile_args)
                model.set_weights(surrogate_models.get_weights())
            del surrogate_models
            gc.collect()
            self.current_epoch = np.zeros(self.surrogate_amount)
        else:
            self.surrogate_models = surrogate_models 
                    
        for model in self.surrogate_models:
            self.reinit_surrogate_model(model)

        self.surrogate_amount = len(self.surrogate_models)
        self.attack_steps = attack_steps
        self.meta_epochs = meta_epochs 
        self.unroll_steps = unroll_steps 
        self.training_args = training_args
        self.adv_learning_rate = adv_learning_rate

        #self._coldstart_surrogate_models()

        # ----------------------------------------------
        # |         Outter loop configurations         |
        # ----------------------------------------------
        # Parameters for crafting additive perturbations
        self.pert_eps = pert_eps
        # Parameters for crafting color perturbations
        self.colorpert_eps = colorpert_eps
        self.colorperturb_smooth_lambda = colorperturb_smooth_lambda
        self.color_gridshape = color_gridshape
        self.target_class = target_class

        self.color_step = 1/100.
        self.perturb_step = 1/255.
        
        # Dataset used for the training
        def _dataset_to_tf(dataset, nameprefix=''):
            x = tf.constant(dataset[0],
                            dtype=TF_DTYPE,
                            name=nameprefix+'_x')
            y = tf.constant(dataset[1],
                            dtype=TF_DTYPE,
                            name=nameprefix+'_y')
            return (x,y)

        self.meta_shadow_dataset = _dataset_to_tf(meta_shadow_dataset,
                                                  'meta_training')
        self.meta_attack_dataset = _dataset_to_tf(meta_attack_dataset,
                                                  'meta_attack')

        self.poison_amount = len(self.meta_attack_dataset[0])

    def _coldstart_surrogate_models(self):
        # Recurrent the current epoch for each surrogate model.
        print("Cold starting surrogate models.")
        for i in range(self.surrogate_amount):
            print("Model {}/{}.".format(i+1, self.surrogate_amount))
            init_epochs = int(np.ceil((1.*(i))/self.surrogate_amount*self.meta_epochs))
            self.surrogate_models[i].fit(self.meta_training_dataset[0],
                                         self.meta_training_dataset[1],
                                         **self.training_args,
                                         epochs=init_epochs)
            self.current_epoch[i] = init_epochs 

    def meta_attack(self):
        # ----------------------------------------------
        # |               Initilization                |
        # ----------------------------------------------
        # Surrogate models
        #self._coldstart_surrogate_models()
        # Attack parameters
        colorperts = tf.Variable(tf.zeros(shape=[self.poison_amount, *self.color_gridshape, 3],
                                                  dtype=TF_DTYPE),
                                 dtype=TF_DTYPE,
                                 name='colorperts')
        _colorgrid = None
        perts = tf.Variable(tf.zeros_like(self.meta_attack_dataset[0], dtype=TF_DTYPE),
                            dtype=TF_DTYPE,
                            name='perts')
        poison_x = tf.Variable(tf.zeros_like(self.meta_attack_dataset[0], dtype=TF_DTYPE),
                               dtype=TF_DTYPE,
                               name='poison_x')

        attack_x = self.meta_attack_dataset[0]
        attack_y = self.meta_attack_dataset[1]

        #colorperts.assign(tf.clip_by_value(colorperts, -self.colorpert_eps, self.colorpert_eps))
        #perts.assign(tf.clip_by_value(perts, -self.pert_eps, self.pert_eps))
        _colorout, _colorgrid = recolor(attack_x, colorperts, grid=_colorgrid)

        poison_y = attack_y

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.adv_learning_rate)

        # For logging
        if self.verbose:
            log_format = """
            ------------------------------
            Attack step: {:3d}/{:3d}
            Overall loss: {:.4f}
            Smooth loss: {:.4f}
            Colorperts: [{:.4f}, {:.4f}]
            Perts: [{:.4f}, {:.4f}]
            Time collapsed: {:.4f}s
            ------------------------------
            """
            
        # ------------------------------------------------
        # |Random sampling meta training and testing data|
        # ------------------------------------------------

        data_augmentation = tf.keras.Sequential(
            [tf.keras.layers.experimental.preprocessing.RandomFlip(),
             tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)]
        )

        self.meta_shadow_dataset = (
            tf.concat([self.meta_shadow_dataset[0],
                       data_augmentation(self.meta_shadow_dataset[0])],
                       axis=0),
            tf.concat([self.meta_shadow_dataset[1],
                       self.meta_shadow_dataset[1]],
                       axis=0)
        )
                                              
        #meta_testing_dataset = (data_augmentation(self.meta_shadow_dataset[0]),
        #                        self.meta_shadow_dataset[1])

        shadow_amount = len(self.meta_shadow_dataset[0])
        split_ratio = 0.5
        meta_training_amount = int(shadow_amount * split_ratio)
        np.random.seed(54321)
        tf.random.set_seed(54321)
        indices = tf.constant(np.arange(0, shadow_amount, dtype=int), dtype=tf.int32)
        meta_training_indices = []
        meta_testing_indices = []

        for _ in range(self.surrogate_amount):
            new_indices = tf.random.shuffle(indices)
            meta_training_indices.append(new_indices[:meta_training_amount])
            meta_testing_indices.append(new_indices[meta_training_amount:])
        
        def _get_splitted_dataset(model_i):
            training_x = tf.gather(self.meta_shadow_dataset[0],
                                   indices=meta_training_indices[model_i],
                                   axis=0)
            training_y = tf.gather(self.meta_shadow_dataset[1],
                                   indices=meta_training_indices[model_i],
                                   axis=0)
            testing_x = tf.gather(self.meta_shadow_dataset[0],
                                   indices=meta_testing_indices[model_i],
                                   axis=0)
            testing_y = tf.gather(self.meta_shadow_dataset[1],
                                   indices=meta_testing_indices[model_i],
                                   axis=0)
            training_dataset = (training_x, training_y)
            testing_dataset = (testing_x, testing_y)
            return (training_dataset, testing_dataset)

        # ----------------------------------------------
        # |    Generating poisons by meta learning     |
        # ----------------------------------------------

        self.meta_model_compile_args = self.model_compile_args.copy()
        self.meta_model_compile_args['loss'] = AdvLoss(target_class=self.target_class)
        self.model_skeleton.compile(**self.meta_model_compile_args)
        if 'xception' in self.model_skeleton.layers[1].name:
            self.model_skeleton.layers[1].trainable = self.fine_tune

        for i in range(self.surrogate_amount):
            self.surrogate_models[i].summary()
        
        for attack_i in range(self.attack_steps):
            start = timeit.default_timer() 
            #poison_x = tf.clip_by_value(_colorout + perts, -1., 1.)
            pert_gradients = []
            colorpert_gradients = []
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                loss_sum = []
                tape.watch(poison_x)
                tape.watch(colorperts)
                # ----------------------------------------------
                # |              Apply perturbs                |
                # ---------------------------------------------- 
                #_colorout, _colorgrid = recolor(attack_x, self.colorpert_eps*tf.tanh(colorperts), grid=_colorgrid)
                #poison_x = tf.clip_by_value(_colorout + self.pert_eps*tf.tanh(perts), 0., 1.) 

                _colorperts = self.colorpert_eps * tf.tanh(colorperts)
                _perts = self.pert_eps * tf.tanh(perts)
                _colorout, _colorgrid = recolor(attack_x, _colorperts, grid=_colorgrid)
                poison_x.assign(tf.clip_by_value(_colorout + _perts, -1., 1.))

                for model_i in range(self.surrogate_amount):
                    meta_training_dataset, meta_testing_dataset = _get_splitted_dataset(model_i)
                    poisoned_y = tf.concat([meta_training_dataset[1], poison_y],0)
                    poisoned_x = tf.concat([meta_training_dataset[0], poison_x],0)

                    # ----------------------------------------------
                    # |         Update surrogate models            |
                    # ----------------------------------------------

                    with tape.stop_recording():
                        if self.current_epoch[model_i] == self.meta_epochs - 1:
                            self.reinit_surrogate_model(self.surrogate_models[model_i])
                            self.current_epoch[model_i] = 0
                        else:
                            self.surrogate_models[model_i].fit(poisoned_x,
                                                               poisoned_y,
                                                               epochs=1,
                                                               **self.training_args)
                            self.current_epoch[model_i] += 1

                    # ----------------------------------------------
                    # |         Unroll surrogate models            |
                    # ----------------------------------------------
                    with tape.stop_recording():
                        #model = tf.keras.models.clone_model(self.surrogate_models[model_i])
                        #model.compile(**self.meta_model_compile_args)
                        # Unroll the surrogate model
                        # Problem: The fit() function won't record the gradient
                        # w.r.t. colorperturbs and perturbs
                        self.model_skeleton.set_weights(self.surrogate_models[model_i].get_weights())
                        print("Unroll surrogate model {}".format(model_i))
                        self.model_skeleton.fit(meta_testing_dataset[0], 
                                                meta_testing_dataset[1],
                                                epochs=self.unroll_steps,
                                                **self.training_args)
                    loss_sum.append(train_loss(poison_y, self.model_skeleton(poison_x)) - \
                                    train_loss(poison_y, self.surrogate_models[model_i](poison_x)))

                    
                smooth_loss = self.colorperturb_smooth_lambda*smoothloss(_colorperts)
                loss_sum.append(smooth_loss)
                loss_sum = tf.reduce_sum(loss_sum)
            colorpert_gradients, pert_gradients = tape.gradient(loss_sum, [colorperts, poison_x])
            gc.collect()

            #colorperts.assign(colorperts-self.color_step*tf.sign(colorpert_gradients))
            #perts.assign(perts-self.perturb_step*tf.sign(pert_gradients))
            # ----------------------------------------------
            # |             Update perturbs                |
            # ----------------------------------------------

            optimizer.apply_gradients(zip([colorpert_gradients, pert_gradients],
                                          [colorperts, perts]))

            #colorperts.assign(tf.clip_by_value(colorperts, -self.colorpert_eps, self.colorpert_eps))
            #perts.assign(tf.clip_by_value(perts, -self.pert_eps, self.pert_eps))

            stop = timeit.default_timer()
            # ----------------------------------------------
            # |                Print logs                  |
            # ----------------------------------------------
            if self.verbose:
                os.system('clear')
                log_str = log_format.format(
                    attack_i+1, self.attack_steps,
                    loss_sum.numpy(),
                    smooth_loss.numpy(),
                    #(self.colorpert_eps*tf.tanh(tf.reduce_min(colorperts))).numpy(),
                    #(self.colorpert_eps*tf.tanh(tf.reduce_max(colorperts))).numpy(),
                    #(self.pert_eps*tf.tanh(tf.reduce_min(perts))).numpy(),
                    #(self.pert_eps*tf.tanh(tf.reduce_max(perts))).numpy(),
                    tf.reduce_min(_colorperts).numpy(),
                    tf.reduce_max(_colorperts).numpy(),
                    tf.reduce_min(_perts).numpy(),
                    tf.reduce_max(_perts).numpy(),
                    stop-start,
                )
                print(log_str)

            # ----------------------------------------------
            # |               Apply perturbs               |
            # ----------------------------------------------

        _colorout, _colorgrid = recolor(attack_x, self.colorpert_eps*tf.tanh(colorperts), grid=_colorgrid)
        poison_x = tf.clip_by_value(_colorout + self.pert_eps*tf.tanh(perts), -1., 1.)

        return (poison_x.numpy(), poison_y.numpy())

    def reinit_surrogate_model(self, model):
        def _reinitialize(model):
        # https://stackoverflow.com/questions/63435679/reset-all-weights-of-keras-model
            for l in model.layers:
                if 'xception' in l.name:
                    l.trainable = self.fine_tune
                if not l.trainable:
                    continue
                if hasattr(l,"kernel_initializer"):
                    try:
                        l.kernel.assign(tf.cast(l.kernel_initializer(tf.shape(l.kernel)),
                                                dtype=l.kernel.dtype))
                    except:
                        print("kernel_initializer")
                        print(l.kernel.dtype)
                        print(tf.cast(l.kernel_initializer(tf.shape(l.kernel)),
                                                dtype=l.kernel.dtype))
                        exit(0)
                if hasattr(l,"bias_initializer"):
                    try:
                        l.bias.assign(tf.cast(l.bias_initializer(tf.shape(l.bias)),
                                            dtype=l.bias.dtype))
                    except:
                        print("bias_initializer")
                        print(l.bias.dtype)
                        print(tf.cast(l.bias_initializer(tf.shape(l.bias)),
                                            dtype=l.bias.dtype).dtype)
                        exit(0)
                if hasattr(l,"recurrent_initializer"):
                    try:
                        l.recurrent_kernel.assign(tf.cast(l.recurrent_initializer(tf.shape(l.recurrent_kernel)),
                                                        l.recurrent_kernel.dtype))
                    except:
                        print('recurrent_initializer')
                        print(l.recurrent_kernel.dtype)
                        print(f.cast(l.recurrent_initializer(tf.shape(l.recurrent_kernel)),
                                                        l.recurrent_kernel.dtype).dtype)
                        exit(0)
        _reinitialize(model)

def test(): 
    IMG_SIZE = [96, 96, 3]
    DATASET_NAME = 'cifar10'
    TARGET_CLASS = 0
    meta_shadow_amount = 9000
    data_range = [0, 9000]

    def build_classifier():
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=tuple(IMG_SIZE)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='tanh'))
        model.add(layers.Dense(10, activation='softmax'))
        return model

    def build_transfer_classifier():
        base_model = FeatureExtractor('vgg16', IMG_SIZE).model
        base_model.trainable = False
        inputs = tf.keras.Input(shape=tuple(IMG_SIZE))
        x = base_model(inputs)
        x = layers.Dense(128, activation='tanh')(x)
        x = layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(inputs, x)
        return model

    model_compile_args = {
        'optimizer':'adam',
        'loss':tf.keras.losses.CategoricalCrossentropy(),
        'metrics':['accuracy']
    }
    training_args={
                    'batch_size': 100,
                }

    def preprocess_fn(x):
        return x / 255.
    
    preprocess_fn = tf.keras.applications.vgg16.preprocess_input
    build_classifier = build_transfer_classifier

    exp_dataset = ExperimentDataset(dataset_name=DATASET_NAME,
                                    preprocess_fn=preprocess_fn,
                                    img_size=IMG_SIZE)
    attack_dataset = exp_dataset.get_attack_dataset()
    meta_shadow_dataset = (attack_dataset[0][:meta_shadow_amount], attack_dataset[1][:meta_shadow_amount])
    meta_attack_dataset = (attack_dataset[0][meta_shadow_amount:], attack_dataset[1][meta_shadow_amount:])
    np.save('attack_x.npy', meta_attack_dataset[0])
    save_sample_img(meta_attack_dataset[0][0], 'attack_sample.png')
    print(meta_attack_dataset[0].shape)
    save_img_from_array(meta_attack_dataset[0][0:10], './attack_samples', preprocess_type='caffe', preprocess_mean=[103.939, 116.779, 123.68])
    clean_model_path = 'cleanmodel.h5'

    if not os.path.exists(clean_model_path):
        clean_model = build_classifier()
        clean_model.compile(**model_compile_args)
        member_dataset = exp_dataset.get_member_dataset(data_range=data_range)
        nonmember_dataset = exp_dataset.get_nonmember_dataset(data_range=data_range)
        clean_model.fit(member_dataset[0], member_dataset[1], **training_args, epochs=20)
        clean_model.save(clean_model_path)
        del clean_model
        tf.keras.backend.clear_session()
        gc.collect()

    if_meta_attack = True 
    dirty_model_path = 'dirtymodel_{}.h5'.format(TARGET_CLASS)
    poison_x_path = 'poison_x_{}.npy'.format(TARGET_CLASS)
    poison_y_path = 'poison_y_{}.npy'.format(TARGET_CLASS)
    if if_meta_attack:
        if os.path.exists(dirty_model_path):
            #dirty_model = tf.keras.models.load_model(dirty_model_path)
            poisons = (np.load(poison_x_path), np.load(poison_y_path))
        else:
            if os.path.exists(poison_x_path):
                poisons = (np.load(poison_x_path), np.load(poison_y_path))
            else:
                meta = MetaAttack(
                            surrogate_models=build_classifier(),
                            meta_shadow_dataset=meta_shadow_dataset,
                            meta_attack_dataset=meta_attack_dataset,
                            target_class=TARGET_CLASS,
                            surrogate_model_pattern='duplicate',
                            surrogate_amount=1,
                            model_compile_args=model_compile_args,
                            training_args={
                                'batch_size': 100,
                            },
                            attack_steps=100,
                            meta_epochs=20,
                            unroll_steps=2,
                            pert_eps=0.08,
                            color_gridshape=[16,32,32],
                            colorpert_eps=0.02,
                            adv_learning_rate=0.001,
                            colorperturb_smooth_lambda=0.1,
                            verbose=True,
                            fine_tune=False,
                            ) 
                poisons = meta.meta_attack()
                np.save(poison_x_path, poisons[0])
                np.save(poison_y_path, poisons[1])

            training_dataset = exp_dataset.get_member_dataset(data_range=data_range)
            poisons = (preprocess_fn(poisons[0]), poisons[1])
            #poisons = (preprocess_fn(((poisons[0]+1)*.5*255).astype(np.uint8)),
            #           poisons[1])
            x = np.r_[training_dataset[0], poisons[0]]
            y = np.r_[training_dataset[1], poisons[1]]
            #save_sample_img(poisons[0][0], 'poison_sample_{}.png'.format(TARGET_CLASS))
            print(np.max(poisons[0]))
            print(np.min(poisons[0]))
            dirty_model = build_classifier()
            dirty_model.compile(**model_compile_args)
            dirty_model.fit(x, y, **training_args, epochs=20)
            dirty_model.save(dirty_model_path)
            del dirty_model
            tf.keras.backend.clear_session()
            gc.collect()

    save_img_from_array(poisons[0][0:10], './poison_samples', preprocess_type='caffe', preprocess_mean=[103.939, 116.779, 123.68])
    dirty_label_model_path = 'dirty_label_model_{}.h5'.format(TARGET_CLASS)

    if os.path.exists(dirty_label_model_path):
        pass
        #dirty_label_model = tf.keras.models.load_model(dirty_label_model_path)
    else:
        dirty_label_model = build_classifier()
        dirty_label_model.compile(**model_compile_args)
        attack_dataset = exp_dataset.get_attack_dataset(target_class=TARGET_CLASS)
        member_dataset = exp_dataset.get_member_dataset(data_range=data_range)
        labels = [_ for _ in range(10)]
        labels.remove(TARGET_CLASS)
        attack_dataset = (attack_dataset[0],
                          tf.keras.utils.to_categorical(
                            np.random.choice(labels, size=len(attack_dataset[0]),replace=True), 
                          num_classes=10))
        history = dirty_label_model.fit(np.r_[member_dataset[0], attack_dataset[0][:600]], 
                              np.r_[member_dataset[1], attack_dataset[1][:600]], **training_args, epochs=20,
                              validation_data=exp_dataset.get_nonmember_dataset(target_class=TARGET_CLASS)) 
        print(history.history['val_accuracy'])
        print(history.history['accuracy'])
        import matplotlib.pyplot as plt
        import pandas as pd
        plt.plot(history.history['val_accuracy'], 'rx-')
        plt.plot(history.history['accuracy'], 'bo-')
        plt.legend(['val_accuracy', 'accuracy'])
        plt.savefig('dirty_label_model_learning_curve_{}.jpg'.format(TARGET_CLASS))
        dirty_label_model.save(dirty_label_model_path)
        del dirty_label_model
        tf.keras.backend.clear_session()
        gc.collect()

    metric = 'Mentr'
    """
    surrogate_model_path = 'surrogate_model_{}.h5'.format(TARGET_CLASS)
    if os.path.exists(surrogate_model_path):
        surrogate_model = tf.keras.models.load_model(surrogate_model_path)
    else:
        surrogate_model = build_classifier()
        surrogate_model.compile(**model_compile_args)
        surrogate_model.fit(np.r_[meta_shadow_dataset[0], poisons[0]],
                            np.r_[meta_shadow_dataset[1], poisons[1]],
                            **training_args, epochs=20)
        surrogate_model.save(surrogate_model_path)
    member_dataset = (meta_shadow_dataset[0][np.argmax(meta_shadow_dataset[1], axis=1)==TARGET_CLASS],
                      meta_shadow_dataset[1][np.argmax(meta_shadow_dataset[1], axis=1)==TARGET_CLASS])
    nonmember_dataset = exp_dataset.get_nonmember_dataset(target_class=TARGET_CLASS)[:len(member_dataset[0])]
    print('surrogate model')

    mia(surrogate_model, member_dataset, nonmember_dataset, metric=metric)
    acc = evaluate_model(surrogate_model, member_dataset)
    print("Training Acc: {}".format(acc))
    acc = evaluate_model(surrogate_model, nonmember_dataset)
    print("Testing Acc: {}".format(acc))
    del surrogate_model
    tf.keras.backend.clear_session()
    gc.collect()
    """

    #member_dataset = exp_dataset.get_member_dataset(data_range=data_range)
    #nonmember_dataset = exp_dataset.get_nonmember_dataset(data_range=data_range)
    for i in range(TARGET_CLASS, TARGET_CLASS+1):
        print('*'*100)
        print(i)
        member_dataset = exp_dataset.get_member_dataset(target_class=i)
        nonmember_dataset = exp_dataset.get_nonmember_dataset(target_class=i)

        member_dataset = (member_dataset[0][:300], member_dataset[1][:300])
        nonmember_dataset = (nonmember_dataset[0][:300], nonmember_dataset[1][:300])

        training_dataset = member_dataset 
        testing_dataset = nonmember_dataset

        metric = 'Mentr'

        print('clean model')
        clean_model = tf.keras.models.load_model(clean_model_path)
        mia(clean_model, member_dataset, nonmember_dataset, metric=metric)
        acc = evaluate_model(clean_model, training_dataset)
        print("Training Acc: {}".format(acc))
        acc = evaluate_model(clean_model, testing_dataset)
        print("Testing Acc: {}".format(acc))
        del clean_model
        tf.keras.backend.clear_session()
        gc.collect()
    
        print('dirty label model')
        dirty_label_model = tf.keras.models.load_model(dirty_label_model_path)
        mia(dirty_label_model, member_dataset, nonmember_dataset, metric=metric)
        acc = evaluate_model(dirty_label_model, training_dataset)
        print("Training Acc: {}".format(acc))
        acc = evaluate_model(dirty_label_model, testing_dataset)
        print("Testing Acc: {}".format(acc))
        del dirty_label_model
        tf.keras.backend.clear_session()
        gc.collect()

        if if_meta_attack:
            print('dirty model')
            dirty_model = tf.keras.models.load_model(dirty_model_path)
            mia(dirty_model, member_dataset, nonmember_dataset, metric=metric) 
            acc = evaluate_model(dirty_model, training_dataset)
            print("Training Acc: {}".format(acc))
            acc = evaluate_model(dirty_model, testing_dataset)
            print("Testing Acc: {}".format(acc))
            del dirty_model
            tf.keras.backend.clear_session()
            gc.collect()

def save_sample_img(x, path):
    x = Image.fromarray((x*255).astype(np.uint8))
    x.save(path)

if __name__ == '__main__':
    try:
        test()
    except Exception as e:
        print(e)
        exit(0)