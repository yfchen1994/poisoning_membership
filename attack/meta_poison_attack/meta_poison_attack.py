import sys
sys.path.append('..')
sys.path.append('.')

import numpy as np
import tensorflow as tf
import gc
from attack.attack_utils import find_nearest_embeedings, sort_best_match_embeeding_heuristis
from .meta_poison_attack import *
from .recolor import *

class MetaAttack:
    def __init__(self,
                 surrogate_models,
                 meta_training_dataset,
                 meta_testing_dataset,
                 meta_attack_dataset,
                 target_class,
                 training_args={
                     'batch_size': 100,
                 },
                 attack_steps=60,
                 meta_epochs=20,
                 unroll_steps=2,
                 pert_eps=0.04,
                 color_gridshape=[16,32,32],
                 colorpert_eps=0.06,
                 adv_learning_rate=0.01,
                 colorperturb_smooth_lambda=0.05,
                 verbose=True,
                 ):
        
        self.verbose = verbose      

        # ----------------------------------------------
        # |       Meta Learning configurations         |
        # ----------------------------------------------
        self.surrogate_models = surrogate_models 
        self.surrogate_amount = len(self.surrogate_models)
        self.attack_steps = attack_steps
        self.meta_epochs = meta_epochs 
        self.unroll_steps = unroll_steps 
        self.training_args = training_argsk

        # ----------------------------------------------
        # |         Outter loop configurations         |
        # ----------------------------------------------
        # Parameters for crafting additive perturbations
        self.pert_eps = pert_eps
        # Parameters for crafting color perturbations
        self.colorpert_eps = colorpert_eps
        self.colorperturb_smooth_lambda = colorperturb_smooth_lambda
        self.color_gridshape = None
        self.target_class = target_class
        
        
        # Dataset used for the training
        def _dataset_to_tf(dataset, nameprefix=''):
            x = tf.constant(dataset[0],
                            dtype=tf.float32,
                            name=nameprefix+'_x')
            y = tf.constant(dataset[1],
                            dtype=tf.float32,
                            name=nameprefix+'_y')
            return (x,y)
        self.meta_training_dataset = _dataset_to_tf(meta_training_dataset,
                                                    'meta_training')
        self.meta_testing_dataset = _dataset_to_tf(meta_testing_dataset,
                                                   'meta_testing')
        self.meta_attack_dataset = _dataset_to_tf(meta_attack_dataset,
                                                  'meta_attack')

        self.poison_amount = len(self.meta_attack_dataset[0])

    def _coldstart_surrogate_models(self):
        # Recurrent the current epoch for each surrogate model.
        self.current_epoch = np.zeros(surrogate_amount)
        for i in range(surrogate_amount):
            init_epochs = np.ceil((1.*(i+1))/surrogate_amount*self.meta_epochs)
            self.surrogate_models[i].fit(self.meta_training_dataset[0],
                                         self.meta_training_dataset[1],
                                         **self.training_args,
                                         epochs=init_epochs)
            self.current_epoch[i] = init_epochs 

    def meta_learning(self):
        # ----------------------------------------------
        # |               Initilization                |
        # ----------------------------------------------
        # Surrogate models
        self._coldstart_surrogate_models()
        # Attack parameters
        colorperts = tf.Variable(tf.zeros([self.poison_amount, *self.color_gridshape, 3]),
                                 dtype=tf.float32,
                                 name='colorperts')
        _colorgrid = None
        perts = tf.Variable(tf.zero_like(self.meta_attack_dataset[0]),
                            dtype=tf.float32,
                            name='perts')
        attack_x = self.meta_attack_dataset[0]
        attack_y = self.meta_attack_dataset[1]

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.adv_learning_rate)

        # For logging
        if self.verbose:
            log_format = """
            ------------------------------
            Attack Step: {:3d}/{:3d}
            Overall loss: {:.4f}
            Overfitting loss: {:.4f}
            Smooth loss: {:.4f}
            ------------------------------
            """

        # ----------------------------------------------
        # |    Generating poisons by meta learning     |
        # ----------------------------------------------
 
        for attack_i in range(self.attack_steps):
            poisoned_x = tf.concat([self.meta_training_dataset[0], attack_x],0)
            poisoned_y = tf.concat([self.meta_training_dataset[1], attack_y],0)
            with tf.GradientTape() as tape:
                loss = tf.zeros(self.surrogate_amount)
                for model_i in range(self.surrogate_amount):
                    model = tf.keras.models.clone_model(self.surrogate_models[model_i])
                    # Unroll the surrogate model
                    with tape.stop_recording():
                        model.fit(poisoned_x, 
                                  poisoned_y, 
                                  epochs=self.unroll_steps,
                                  **self.training_args)
                    loss[model_i] = adv_loss(self.meta_testing_dataset[1],
                                                model(self.meta_testing_dataset[0],
                                                target_class=self.target_class))
                    # Advance epoch
                    with tape.stop_recording():
                        if self.current_epoch[i] == self.meta_epochs - 1:
                            self.reinit_surrogate_model(self.surrogate_models[model_i])
                        else:
                            self.surrogate_models[model_i].fit(self.meta_training_dataset[0],
                                                            self.meta_training_dataset[1],
                                                            epochs=1,
                                                            **self.training_args)
                    smooth_loss = smoothloss(colorperts)
                overfit_loss = tf.reduce_mean(loss) 
                loss_sum = overfit_loss + self.colorperturb_smooth_lambda*smooth_loss
            grads = tape.gradient(loss_sum, [perts, colorperts])
            optimizer.apply_gradients(zip(grads, [perts, colorperts]))

            # ----------------------------------------------
            # |                Print logs                  |
            # ----------------------------------------------
            if self.verbose:
                os.system('clear')
                log_str = log_format.format(
                    attack_i+1, self.attack_steps,
                    loss_sum.numpy(),
                    overfit_loss.numpy(),
                    smooth_loss.numpy()
                )
                print(log_str)
            

            # ----------------------------------------------
            # |               Apply perturbs               |
            # ----------------------------------------------
            colorperts = tf.clip_by_value(colorperts, -self.colorpert_eps, self.colorpert_eps)
            perts = tf.clip_by_value(perts, -self.pert_eps, self.pert_eps)
            attack_x, _colorgrid = recolor(attack_x, colorperts, grid=_colorgrid)
            attack_x = attack_x + perts
            attack_x = tf.clip_by_value(attack_x, 0., 1.)
        return (attack_x.numpy(), attack_y.numpy())


    def reinit_surrogate_model(self, model):
        def _reinitialize(model):
        # https://stackoverflow.com/questions/63435679/reset-all-weights-of-keras-model
            for l in model.layers:
                if hasattr(l,"kernel_initializer"):
                    l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
                if hasattr(l,"bias_initializer"):
                    l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
                if hasattr(l,"recurrent_initializer"):
                    l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
        _reinitialize(model)

    def update_model(self):
        pass

    def update_perturb(self):

    def pgd(self):
        pass
