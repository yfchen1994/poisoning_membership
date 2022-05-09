import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
PROJ_DIR = '../../'

import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)

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
                 attack_steps=5,
                 attack_current_step=0,
                 meta_epochs=20,
                 unroll_steps=2,
                 pert_eps=0.08,
                 color_gridshape=[16,32,32],
                 colorpert_eps=0.04,
                 adv_learning_rate=0.001,
                 colorperturb_smooth_lambda=0.05,
                 verbose=True,
                 colorperts=None,
                 perts=None,
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

        if 'xception' in self.model_skeleton.layers[1].name:
            self.model_skeleton.layers[1].trainable = self.fine_tune

                    
        self.attack_steps = attack_steps
        self.meta_epochs = meta_epochs 
        self.unroll_steps = unroll_steps 
        self.training_args = training_args
        self.adv_learning_rate = adv_learning_rate

        self.attack_current_step = attack_current_step

        self.colorperts = colorperts
        self.perts = perts

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

        for model_i in range(self.surrogate_amount):
            print("Model :{}, {:.4}".format(model_i, 
                                            evaluate_model(self.surrogate_models[model_i], self.meta_shadow_dataset)))

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
            self.current_iter[i] = init_epochs 

    
    def meta_attack(self):
        # ----------------------------------------------
        # |               Initilization                |
        # ----------------------------------------------
        # Surrogate models
        #self._coldstart_surrogate_models()
        # Attack parameters

        @tf.function
        def train_minibatch(model, dataset, optimizer, loss_fn):
            x, y = dataset
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model.trainable_variables)
                prediction = model(x)
                loss = loss_fn(prediction, y)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))


        if self.colorperts is None:
            colorperts = tf.Variable(tf.zeros(shape=[self.poison_amount, *self.color_gridshape, 3],
                                                     dtype=TF_DTYPE),
                                    dtype=TF_DTYPE,
                                    name='colorperts')
        else:
            colorperts = tf.Variable(self.colorperts,
                                     dtype=TF_DTYPE,
                                     name='colorperts')
        _colorgrid = None

        if self.perts is None:
            perts = tf.Variable(tf.zeros_like(self.meta_attack_dataset[0], dtype=TF_DTYPE),
                                dtype=TF_DTYPE,
                                name='perts')
        else:
            perts = tf.Variable(self.perts,
                                dtype=TF_DTYPE,
                                name='perts')

        attack_x = tf.Variable(self.meta_attack_dataset[0], dtype=TF_DTYPE, trainable=False)
        attack_y = self.meta_attack_dataset[1]

        #colorperts.assign(tf.clip_by_value(colorperts, -self.colorpert_eps, self.colorpert_eps))
        #perts.assign(tf.clip_by_value(perts, -self.pert_eps, self.pert_eps))

        optimizer_colorpert = tf.keras.optimizers.Adam(learning_rate=self.adv_learning_rate)
        optimizer_pert = tf.keras.optimizers.Adam(learning_rate=5*self.adv_learning_rate)

        # For logging
        if self.verbose:
            log_format = """
            ------------------------------
            Attack Cycle: {:2d}
            Epoch: {:2d}/{:2d}
            Batch: {:5d}
            Overall loss: {:.8f}
            Smooth loss: {:.8f}
            Colorperts: [{:.6f}, {:.6f}]
            Perts: [{:.6f}, {:.6f}]
            Time collapsed: {:.4f}s
            ------------------------------
            """
            
        # ------------------------------------------------
        # |Random sampling meta training and testing data|
        # ------------------------------------------------
        meta_shadow_amount = len(self.meta_shadow_dataset[0])
        split_ratio = 0.5
        meta_training_amount = int(meta_shadow_amount * split_ratio)

        indices = tf.constant(np.arange(0, meta_shadow_amount, dtype=int), dtype=tf.int32)
        meta_training_indices = []
        meta_testing_indices = []
        
        for _ in range(self.surrogate_amount):
            new_indices = tf.random.shuffle(indices)
            meta_training_indices.append(new_indices[:meta_training_amount])
            meta_testing_indices.append(new_indices[meta_training_amount:])
        
        @tf.function
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

        #@tf.function
        def _create_dataset_iterator(dataset):
            dataset = tf.data.Dataset.from_tensor_slices(dataset)
            dataset_tmp = dataset.shuffle(buffer_size=1024).repeat()\
                            .batch(self.training_args['batch_size'])
            return iter(dataset_tmp)

        meta_training_dsi = []
        meta_testing_dsi = []   

        for model_i in range(self.surrogate_amount):
            training_dataset, testing_dataset = _get_splitted_dataset(model_i) 
            meta_training_dsi.append(_create_dataset_iterator(training_dataset))
            meta_testing_dsi.append(_create_dataset_iterator(testing_dataset))

        @tf.function
        def _get_dataset_batch(batchdataset_iterator):
            return batchdataset_iterator.get_next()

        def _update_variable(variable, idxes, values):
            def _assign_variable(arg): 
                idx, value = arg
                variable[idx].assign(value)
                return True

            @tf.function
            def _func(elements):
                return tf.map_fn(_assign_variable,
                                elements,
                                dtype=tf.bool,
                                parallel_iterations=500)
            _func((idxes, values))

        adv_loss = AdvLoss(target_class=self.target_class)
        train_loss = tf.keras.losses.CategoricalCrossentropy()

        @tf.function
        def _get_update_gradients(model_i, 
                                  attack_x_minibatch,
                                  perts_minibatch, 
                                  colorperts_minibatch,
                                  _colorgrid=None):
            perts_minibatch = tf.Variable(perts_minibatch, dtype=TF_DTYPE)
            colorperts_minibatch = tf.Variable(colorperts_minibatch, dtype=TF_DTYPE)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(perts_minibatch)
                tape.watch(colorperts_minibatch)
                _colorout, _colorgrid = recolor(attack_x_minibatch, 
                                                colorperts_minibatch, 
                                                grid=_colorgrid)
            
                poison_x = tf.clip_by_value(_colorout + perts_minibatch, 0., 1.) 
                loss = train_loss(poison_y, self.model_skeleton(poison_x)) - \
                        train_loss(poison_y, self.surrogate_models[model_i](poison_x))
            _color_grad, _pert_grad = tape.gradient(loss, 
                                                    [colorperts_minibatch, perts_minibatch])
            return (_color_grad, _pert_grad, poison_x)

        def calculate_perturb():
            meta_training_dataset = _get_dataset_batch(meta_training_dsi[model_i])
            meta_testing_dataset = _get_dataset_batch(meta_testing_dsi[model_i])
            poisoned_y = tf.concat([meta_training_dataset[1], poison_y], 0)
            poisoned_x = tf.concat([meta_training_dataset[0], poison_x], 0)

            _b = timeit.default_timer()

            # ----------------------------------------------
            # |         Update surrogate models            |
            # ----------------------------------------------
            train_minibatch(self.surrogate_models[model_i],
                            (poisoned_x, poisoned_y),
                            optimizer=optimizer_learning[model_i],
                            loss_fn=train_loss)

            # ----------------------------------------------
            # |         Unroll surrogate models            |
            # ----------------------------------------------

            train_minibatch(self.model_skeleton,
                            meta_testing_dataset,
                            optimizer=optimizer_learning_unroll[model_i],
                            loss_fn=adv_loss)
            stop += (_b - _a)
            
            loss = train_loss(poison_y, self.model_skeleton(poison_x)) - \
                train_loss(poison_y, self.surrogate_models[model_i](poison_x))
            loss_cached.append(loss)

        # dataset iterator 
        """
        meta_training_dsi = []
        meta_testing_dsi = []
        for model_i in range(self.surrogate_amount):
            training_dataset, testing_dataset = _get_splitted_dataset(model_i)

            training_dataset = tf.data.Dataset.from_tensor_slices(training_dataset)
            testing_dataset = tf.data.Dataset.from_tensor_slices(testing_dataset)

            training_dataset = training_dataset.shuffle(buffer_size=1024).repeat()\
                               .batch(self.training_args['batch_size']) 
            testing_dataset = testing_dataset.shuffle(buffer_size=1024).repeat()\
                               .batch(self.training_args['batch_size']) 

            meta_training_dsi.append(iter(training_dataset))
            meta_testing_dsi.append(iter(testing_dataset))
        del training_dataset, testing_dataset
        gc.collect()
        """

        attack_batch_size = int(len(self.meta_attack_dataset[0])/(split_ratio*meta_shadow_amount)*self.training_args['batch_size'])

        # ----------------------------------------------
        # |    Generating poisons by meta learning     |
        # ----------------------------------------------

        #print("This part will improve other part's")
        for model_i in range(self.surrogate_amount):
            #start = timeit.default_timer()
            meta_training_dataset = _get_dataset_batch(meta_training_dsi[model_i])
            meta_testing_dataset = _get_dataset_batch(meta_testing_dsi[model_i])
            #stop = timeit.default_timer()
            #print(stop-start)

        self.meta_model_compile_args = self.model_compile_args.copy()

        for attack_i in range(self.attack_current_step, self.attack_current_step+self.attack_steps):
            optimizer_learning = [tf.keras.optimizers.Adam() for _ in range(self.surrogate_amount)]
            optimizer_learning_unroll = [tf.keras.optimizers.Adam() for _ in range(self.surrogate_amount)]
            gc.collect()

            for epoch in range(self.meta_epochs):
                attack_indices = iter(tf.data.Dataset.range(len(self.meta_attack_dataset[0]))\
                                  .shuffle(buffer_size=1024).batch(attack_batch_size))
                iters = 0
                while True:
                    iters += 1
                    # Select vars
                    try:
                        attack_idx = next(attack_indices)
                    except StopIteration:
                        break

                    perts_minibatch = tf.Variable(tf.gather(perts, indices=attack_idx, axis=0),
                                                  dtype=TF_DTYPE)
                    colorperts_minibatch = tf.Variable(tf.gather(colorperts, indices=attack_idx, axis=0),
                                                       dtype=TF_DTYPE)
                    attack_x_minibatch = tf.gather(attack_x, indices=attack_idx,
                                                   axis=0)
                    poison_y = tf.gather(attack_y, indices=attack_idx,
                                         axis=0)

                    color_grads_cached = []
                    pert_grads_cached = []
                    loss_sum = 0
                    loss_cached = []
                    start = timeit.default_timer()
                   
                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                        tape.watch(perts_minibatch)
                        tape.watch(colorperts_minibatch)

                        _colorperts = self.colorpert_eps*tf.tanh(colorperts_minibatch)
                        _perts = 2*self.pert_eps*tf.tanh(perts_minibatch)
                        
                        _colorout, _colorgrid = recolor(attack_x_minibatch, 
                                                        _colorperts, 
                                                        grid=_colorgrid)    

                        poison_x = tf.clip_by_value(_colorout + _perts, -1., 1.) 
                        loss_cached = []
                   
                        for model_i in range(self.surrogate_amount):
                            # ----------------------------------------------
                            # |         Update surrogate models            |
                            # ----------------------------------------------
                            poisoned_y = tf.concat([meta_training_dataset[1], poison_y], 0)
                            poisoned_x = tf.concat([meta_training_dataset[0], poison_x], 0)

                            with tape.stop_recording():
                                meta_training_dataset = _get_dataset_batch(meta_training_dsi[model_i])
                                train_minibatch(self.surrogate_models[model_i],
                                                (poisoned_x, poisoned_y),
                                                optimizer=optimizer_learning[model_i],
                                                loss_fn=train_loss)
 
                            # ----------------------------------------------
                            # |         Unroll surrogate models            |
                            # ----------------------------------------------
                            with tape.stop_recording():
                                meta_testing_dataset = _get_dataset_batch(meta_testing_dsi[model_i])
                                self.model_skeleton.set_weights(self.surrogate_models[model_i].get_weights())
                                train_minibatch(self.model_skeleton,
                                                meta_testing_dataset,
                                                optimizer=optimizer_learning_unroll[model_i],
                                                loss_fn=adv_loss)
                            
                            loss = train_loss(poisoned_y, self.model_skeleton(poisoned_x)) - \
                                    train_loss(poisoned_y, self.surrogate_models[model_i](poisoned_x))
                            loss_cached.append(loss)

                        smooth_loss = self.colorperturb_smooth_lambda*smoothloss(_colorperts)
                        loss_cached.append(smooth_loss)
                        loss = tf.reduce_sum(loss_cached)
                    color_grads, pert_grads = tape.gradient(loss,
                                                            [colorperts_minibatch, perts_minibatch])

                    optimizer_colorpert.apply_gradients([(color_grads, colorperts_minibatch)])
                    optimizer_pert.apply_gradients([(pert_grads, perts_minibatch)])
                    #colorperts_minibatch.assign(colorperts_minibatch-self.adv_learning_rate*tf.sign(color_grads))
                    #perts_minibatch.assign(perts_minibatch-self.adv_learning_rate*tf.sign(pert_grads))

                    # ----------------------------------------------
                    # |               Apply perturbs               |
                    # ----------------------------------------------

                    # ----------------------------------------------
                    # |            Update poisons                  |
                    # ----------------------------------------------
                    """
                    _update_variable(colorperts, 
                                     attack_idx, 
                                     tf.clip_by_value(colorperts_minibatch, 
                                                      -self.colorpert_eps, self.colorpert_eps))
                    _update_variable(perts, 
                                     attack_idx, 
                                     tf.clip_by_value(perts_minibatch, 
                                                      -self.pert_eps, self.pert_eps))
                    """
                    _update_variable(colorperts, attack_idx, colorperts_minibatch)
                    _update_variable(perts, attack_idx, perts_minibatch)

                    stop = timeit.default_timer()
                    # ----------------------------------------------
                    # |                Print logs                  |
                    # ----------------------------------------------
                    if self.verbose:
                        os.system('clear')
                        log_str = log_format.format(
                            attack_i+1,
                            epoch+1, self.meta_epochs,
                            iters,
                            loss.numpy(),
                            smooth_loss.numpy(),
                            tf.reduce_min(_colorperts).numpy(),
                            tf.reduce_max(_colorperts).numpy(),
                            tf.reduce_min(_perts).numpy(),
                            tf.reduce_max(_perts).numpy(),
                            stop-start,
                        )
                        print(log_str)
                
                for model_i in range(self.surrogate_amount):
                    print("Model :{}, {:.4}".format(model_i, 
                                                    evaluate_model(self.surrogate_models[model_i], self.meta_shadow_dataset)))

            # Reinitialize the surrogate model
            for model_i in range(self.surrogate_amount):
                self.reinit_surrogate_model(self.surrogate_models[model_i])

        _colorout, _colorgrid = recolor(attack_x, self.colorpert_eps*tf.tanh(colorperts))
        attack_x.assign(tf.clip_by_value(_colorout+2*self.pert_eps*tf.tanh(perts), -1., 1.))

        return (attack_x.numpy(), attack_y.numpy()), colorperts.numpy(), perts.numpy()


    def reinit_surrogate_model(self, model):
        def _reinitialize(model):
        # https://stackoverflow.com/questions/63435679/reset-all-weights-of-keras-model
            for l in model.layers:
                if 'xception' in l.name:
                    l.trainable = self.fine_tune
            for l in model.layers:
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
    DATASET_NAME = 'stl10'
    IMG_SIZE = [96, 96, 3]
    data_range = [0, 4000]
    meta_shadow_amount = 3600

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

    def build_xception_classifier():
        base_model = tf.keras.applications.Xception(
            weights='imagenet',
            input_shape=tuple(IMG_SIZE),
            include_top=False,
        )
        base_model.trainable = False
        inputs = tf.keras.Input(shape=tuple(IMG_SIZE))
        x = base_model(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='tanh')(x)
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

    preprocess_fn = tf.keras.applications.xception.preprocess_input
    build_classifier = build_xception_classifier

    exp_dataset = ExperimentDataset(dataset_name=DATASET_NAME,
                                    preprocess_fn=preprocess_fn,
                                    img_size=IMG_SIZE)
    attack_dataset = exp_dataset.get_attack_dataset()
    meta_shadow_dataset = (attack_dataset[0][:meta_shadow_amount], attack_dataset[1][:meta_shadow_amount])

    data_augmentation = tf.keras.Sequential(
            [tf.keras.layers.experimental.preprocessing.RandomFlip(),
             tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)]
        )
    meta_shadow_dataset = (np.r_[meta_shadow_dataset[0],
                                 data_augmentation(meta_shadow_dataset[0]).numpy()],
                           np.r_[meta_shadow_dataset[1],
                                 meta_shadow_dataset[1]])
    print(meta_shadow_dataset[0].shape)
    print(meta_shadow_dataset[1].shape)
    meta_attack_dataset = (attack_dataset[0][meta_shadow_amount:], attack_dataset[1][meta_shadow_amount:])
    np.save('attack_x.npy', meta_attack_dataset[0])
    save_sample_img(meta_attack_dataset[0][0], 'attack_sample.png')
    save_img_from_array(meta_attack_dataset[0][0:10], './attack_samples')
    save_img_from_array(meta_shadow_dataset[0][3600:3610], './shadow_samples')
    print("????????")

    if_meta_attack = True
    dirty_model_path = 'dirtymodel.h5'
    if if_meta_attack:
        if os.path.exists(dirty_model_path):
            dirty_model = tf.keras.models.load_model(dirty_model_path)
            poisons = (np.load('poison_x.npy'), np.load('poison_y.npy'))
        else:
            if os.path.exists('poison_x.npy'):
                poisons = (np.load('poison_x.npy'), np.load('poison_y.npy'))
            else:
                colorperts = None
                perts = None
                for i in range(2):
                    meta = MetaAttack(
                                surrogate_models=build_classifier(),
                                meta_shadow_dataset=meta_shadow_dataset,
                                meta_attack_dataset=meta_attack_dataset,
                                target_class=0,
                                surrogate_model_pattern='duplicate',
                                surrogate_amount=2,
                                model_compile_args=model_compile_args,
                                training_args={
                                    'batch_size': 100,
                                },
                                attack_steps=1,
                                attack_current_step=i,
                                meta_epochs=20,
                                unroll_steps=1,
                                pert_eps=0.08,
                                color_gridshape=[16,32,32],
                                colorpert_eps=0.04,
                                adv_learning_rate=10,
                                colorperturb_smooth_lambda=0.05,
                                verbose=True,
                                colorperts=colorperts,
                                perts=perts,
                                ) 
                    poisons, colorperts, perts = meta.meta_attack()
                    save_sample_img(poisons[0][0], 'poison_sample.png')
                    np.save('poison_x.npy', poisons[0])
                    np.save('poison_y.npy', poisons[1])
                    del meta
                    tf.keras.backend.clear_session()
                    gc.collect()

                    save_img_from_array(poisons[0][10:20], './poison_samples')
                    save_sample_img(poisons[0][0], 'poison_sample.png')

            training_dataset = exp_dataset.get_member_dataset(data_range=data_range)
            #x = np.r_[training_dataset[0], (poisons[0]*255).astype(np.uint8)/255.]
            x = np.r_[training_dataset[0], ((poisons[0]+1)*0.5*255).astype(np.uint8)/255.*2-1]
            y = np.r_[training_dataset[1], poisons[1]]

            dirty_model = build_classifier()
            dirty_model.compile(**model_compile_args)
            dirty_model.fit(x, y, **training_args, epochs=20)
            dirty_model.save(dirty_model_path)

        save_img_from_array(poisons[0][10:20], './poison_samples')
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
        attack_dataset = exp_dataset.get_attack_dataset(target_class=0)
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
    np.random.seed(54321)
    tf.random.set_seed(54321)
    test()