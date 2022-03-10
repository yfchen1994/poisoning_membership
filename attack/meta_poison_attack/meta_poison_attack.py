import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
PROJ_DIR = '../../'

import numpy as np
import tensorflow as tf
TF_DTYPE = tf.float16
if TF_DTYPE is tf.float16:
    print("Setting float16 precision")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.keras.backend.set_floatx('float16')
import gc
from recolor import *
sys.path.append(PROJ_DIR)
from models.build_models import *
from attack.attack_utils import mia, evaluate_model
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
                 ):
        
        self.verbose = verbose      

        # ----------------------------------------------
        # |       Meta Learning configurations         |
        # ----------------------------------------------
        self.model_compile_args = model_compile_args
        self.surrogate_amount = surrogate_amount
        if surrogate_model_pattern == 'duplicate':
            self.surrogate_models = [
                tf.keras.models.clone_model(surrogate_models) 
                for _ in range(surrogate_amount)
                ]
            self.model_skeleton = tf.keras.models.clone_model(surrogate_models)
            del surrogate_models
            gc.collect()
            for model in self.surrogate_models:
                model.compile(**model_compile_args)
                self.reinit_surrogate_model(model)
            self.current_epoch = np.zeros(self.surrogate_amount)
        else:
            self.surrogate_models = surrogate_models 
                    
        self.surrogate_amount = len(self.surrogate_models)
        self.attack_steps = attack_steps
        self.meta_epochs = meta_epochs 
        self.unroll_steps = unroll_steps 
        self.training_args = training_args
        self.adv_learning_rate = adv_learning_rate

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
        colorperts = tf.Variable(tf.random.normal(shape=[self.poison_amount, *self.color_gridshape, 3],
                                                  dtype=TF_DTYPE),
                                 dtype=TF_DTYPE,
                                 name='colorperts')
        _colorgrid = None
        perts = tf.Variable(tf.zeros_like(self.meta_attack_dataset[0], dtype=TF_DTYPE),
                            dtype=TF_DTYPE,
                            name='perts')

        attack_x = self.meta_attack_dataset[0]
        attack_y = self.meta_attack_dataset[1]

        colorperts.assign(tf.clip_by_value(colorperts, -self.colorpert_eps, self.colorpert_eps))
        perts.assign(tf.clip_by_value(perts, -self.pert_eps, self.pert_eps))
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
        shadow_amount = len(self.meta_shadow_dataset[0])
        split_ratio = 0.7
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
        for attack_i in range(self.attack_steps):
            start = timeit.default_timer() 
            poison_x = tf.clip_by_value(_colorout + perts, 0., 1.)
            pert_gradients = []
            colorpert_gradients = []
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                loss_sum = []
                tape.watch(perts)
                tape.watch(colorperts)
                _colorout, _colorgrid = recolor(attack_x, colorperts, grid=_colorgrid)
                poison_x = tf.clip_by_value(_colorout + perts, 0., 1.) 
                for model_i in range(self.surrogate_amount):
                    meta_training_dataset, meta_testing_dataset = _get_splitted_dataset(model_i)
                    poisoned_y = tf.concat([meta_training_dataset[1], poison_y],0)
                    poisoned_x = tf.concat([meta_training_dataset[0], poison_x],0)
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

                    
                smooth_loss = self.colorperturb_smooth_lambda*smoothloss(colorperts)
                loss_sum.append(smooth_loss)
                loss_sum = tf.reduce_mean(loss_sum)
            colorpert_gradients, pert_gradients = tape.gradient(loss_sum, [colorperts, perts])
            del model
            gc.collect()

            #colorperts.assign(colorperts-self.color_step*tf.sign(colorpert_gradients))
            #perts.assign(perts-self.perturb_step*tf.sign(pert_gradients))
            optimizer.apply_gradients(zip([colorpert_gradients, pert_gradients],
                                          [colorperts, perts]))

            # ----------------------------------------------
            # |               Apply perturbs               |
            # ----------------------------------------------

            colorperts.assign(tf.clip_by_value(colorperts, -self.colorpert_eps, self.colorpert_eps))
            perts.assign(tf.clip_by_value(perts, -self.pert_eps, self.pert_eps))
            _colorout, _colorgrid = recolor(attack_x, colorperts, grid=_colorgrid)
            poison_x = tf.clip_by_value(_colorout + perts, 0., 1.)

            # ----------------------------------------------
            # |         Update surrogate models            |
            # ----------------------------------------------

            for model_i in range(self.surrogate_amount):
                if self.current_epoch[model_i] == self.meta_epochs - 1:
                    self.reinit_surrogate_model(self.surrogate_models[model_i])
                    self.current_epoch[model_i] = 0
                else:
                    self.surrogate_models[model_i].fit(poisoned_x,
                                                       poisoned_y,
                                                       epochs=1,
                                                       **self.training_args)
                    self.current_epoch[model_i] += 1
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
                    tf.reduce_min(colorperts).numpy(),
                    tf.reduce_max(colorperts).numpy(),
                    tf.reduce_min(perts).numpy(),
                    tf.reduce_max(perts).numpy(),
                    stop-start,
                )
                print(log_str)

            

        return (poison_x.numpy(), poison_y.numpy())


    def reinit_surrogate_model(self, model):
        def _reinitialize(model):
        # https://stackoverflow.com/questions/63435679/reset-all-weights-of-keras-model
            for l in model.layers:
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

    def build_classifier():
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='tanh'))
        model.add(layers.Dense(10, activation='softmax'))
        return model

    def build_xception_classifier():
        base_model = tf.keras.applications.Xception(
            weights='imagenet',
            input_shape=(224,224,3),
            include_top=False,
        )
        inputs = tf.keras.Input(shape=(224,224,3))

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

    exp_dataset = ExperimentDataset(dataset_name='stl10',
                                    preprocess_fn=preprocess_fn,
                                    img_size=[96,96,3])
    attack_dataset = exp_dataset.get_attack_dataset()
    meta_shadow_amount = 3600
    meta_shadow_dataset = (attack_dataset[0][:meta_shadow_amount], attack_dataset[1][:meta_shadow_amount])
    meta_attack_dataset = (attack_dataset[0][meta_shadow_amount:], attack_dataset[1][meta_shadow_amount:])
    np.save('attack_x.npy', meta_attack_dataset[0])
    save_sample_img(meta_attack_dataset[0][0], 'attack_sample.png')

    data_range = [0, 4000]
    if_meta_attack = True 
    dirty_model_path = 'dirtymodel.h5'
    if if_meta_attack:
        if os.path.exists(dirty_model_path):
            dirty_model = tf.keras.models.load_model(dirty_model_path)
            poisons = (np.load('poison_x.npy'), np.load('poison_y.npy'))
        else:
            meta = MetaAttack(
                        surrogate_models=build_classifier(),
                        meta_shadow_dataset=meta_shadow_dataset,
                        meta_attack_dataset=meta_attack_dataset,
                        target_class=0,
                        surrogate_model_pattern='duplicate',
                        surrogate_amount=12,
                        model_compile_args=model_compile_args,
                        training_args={
                            'batch_size': 100,
                        },
                        attack_steps=100,
                        meta_epochs=20,
                        unroll_steps=1,
                        pert_eps=0.08,
                        color_gridshape=[16,32,32],
                        colorpert_eps=0.02,
                        adv_learning_rate=0.01,
                        colorperturb_smooth_lambda=0.05,
                        verbose=True,
                        ) 
            poisons = meta.meta_attack()
            dirty_model = build_classifier()
            training_dataset = exp_dataset.get_member_dataset(data_range=data_range)
            x = np.r_[training_dataset[0], poisons[0]]
            y = np.r_[training_dataset[1], poisons[1]]
            np.save('poison_x.npy', poisons[0])
            np.save('poison_y.npy', poisons[1])
            dirty_model.compile(**model_compile_args)
            dirty_model.fit(x, y, **training_args, epochs=20)
            dirty_model.save(dirty_model_path)

        save_sample_img(poisons[0][0], 'poison_sample.png')



    clean_model_path = 'cleanmodel.h5'

    if os.path.exists(clean_model_path):
        clean_model = tf.keras.models.load_model(clean_model_path)
    else:
        clean_model = build_classifier()
        clean_model.compile(**model_compile_args)
        member_dataset = exp_dataset.get_member_dataset(data_range=data_range)
        nonmember_dataset = exp_dataset.get_nonmember_dataset(data_range=data_range)
        clean_model.fit(member_dataset[0], member_dataset[1], **training_args, epochs=20)
        clean_model.save(clean_model_path)


    dirty_label_model_path = 'dirty_label_model.h5'

    if os.path.exists(dirty_label_model_path):
        dirty_label_model = tf.keras.models.load_model(dirty_label_model_path)
    else:
        dirty_label_model = build_classifier()
        dirty_label_model.compile(**model_compile_args)
        attack_data = exp_dataset.get_attack_dataset(target_class=0)
        member_dataset = exp_dataset.get_member_dataset(data_range=data_range)
        attack_dataset = (attack_dataset[0],
                          tf.keras.utils.to_categorical(np.random.random_integers(1,9,len(attack_dataset[1])), num_classes=10))
        dirty_label_model.fit(np.r_[member_dataset[0], attack_dataset[0]], 
                              np.r_[member_dataset[1], attack_dataset[1]], **training_args, epochs=20) 
        dirty_label_model.save(dirty_label_model_path)

    member_dataset = exp_dataset.get_member_dataset(data_range=data_range, target_class=0)
    nonmember_dataset = exp_dataset.get_nonmember_dataset(data_range=data_range, target_class=0)
    training_dataset = exp_dataset.get_member_dataset()
    testing_dataset = exp_dataset.get_nonmember_dataset()

    metric = 'Mentr'

    print('clean model')
    mia(clean_model, member_dataset, nonmember_dataset, metric=metric)
    acc = evaluate_model(clean_model, training_dataset)
    print("Training Acc: {}".format(acc))
    acc = evaluate_model(clean_model, testing_dataset)
    print("Testing Acc: {}".format(acc))


   
    print('dirty label model')
    mia(dirty_label_model, member_dataset, nonmember_dataset, metric=metric)
    acc = evaluate_model(dirty_label_model, training_dataset)
    print("Training Acc: {}".format(acc))
    acc = evaluate_model(dirty_label_model, testing_dataset)
    print("Testing Acc: {}".format(acc))


    if if_meta_attack:
        print('dirty model')
        mia(dirty_model, member_dataset, nonmember_dataset, metric=metric) 
        acc = evaluate_model(dirty_model, training_dataset)
        print("Training Acc: {}".format(acc))
        acc = evaluate_model(dirty_model, testing_dataset)
        print("Testing Acc: {}".format(acc))


def save_sample_img(x, path):
    x = Image.fromarray((x*255).astype(np.uint8))
    x.save(path)

if __name__ == '__main__':
    try:
        test()
    except Exception as e:
        print(e)
        exit(0)