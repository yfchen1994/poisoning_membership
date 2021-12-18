import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf
import gc
from attack.attack_utils import find_nearest_embeedings, sort_best_match_embeeding_heuristis

def clean_label_attack(encoder,
                       target_class,
                       attack_dataset,
                       seed_amount=0,
                       anchorpoint_amount=0,
                       poison_config=None,
                       image_scale=np.array([[-1., 1.] for i in range(3)]).transpose()):
    """ The clean label attack.

    Args:
        encoder (keras model): The pretrained feature extractors.
        target_class (int): The subclass that the attacker aims to mount poisoning.
        attack_dataset (dataset tuple): The attacker's dataset.
        seed_amount (int, optional): The amount of the seed images. Defaults to 0.
        anchorpoint_amount (int, optional): The amount of the anchorpoint images. Defaults to 0.
        poison_config (dict, optional): Parameters used to craft poisons. Defaults to None.

    Returns:
        dataset tuple: The poisoning dataset.
    """

    class_label = np.argmax(attack_dataset[1], axis=1)
    seed_x = attack_dataset[0][np.where(class_label != target_class)]
    seed_y = attack_dataset[1][np.where(class_label != target_class)]

    if seed_amount <= 0:
        seed_amount = len(seed_x)
    if seed_amount < len(seed_x):
        seed_x = seed_x[:seed_amount]
        seed_y = seed_y[:seed_amount]
    seed_dataset = (seed_x, seed_y)
    del seed_x, seed_y

    anchorpoint_x = attack_dataset[0][np.where(class_label == target_class)]
    anchorpoint_y = attack_dataset[1][np.where(class_label == target_class)]

    if anchorpoint_amount <= 0:
        anchorpoint_amount = len(anchorpoint_x)
    if anchorpoint_amount < len(anchorpoint_x):
        anchorpoint_x = anchorpoint_x[:anchorpoint_amount]
        anchorpoint_y = anchorpoint_y[:anchorpoint_amount]
    anchorpoint_dataset = (anchorpoint_x, anchorpoint_y)
    del anchorpoint_x, anchorpoint_y

    poison_dataset, selected_anchorpoint_dataset, seed_idx = craft_poisons(encoder,
                                                                           seed_dataset,
                                                                           anchorpoint_dataset,
                                                                           image_scale=image_scale,
                                                                           **poison_config)

    return (poison_dataset, selected_anchorpoint_dataset, seed_idx)

def craft_poisons(encoder,
                  seed_dataset,
                  anchorpoint_dataset,
                  learning_rate=0.001,
                  batch_size=16,
                  iters=1000,
                  image_scale=np.array([[-1., 1.] for i in range(3)]).transpose(),
                  if_selection=True):

    """ Crafting poisons.

    Args:
        encoder (keras model): The pretrained feature extractors.
        seed_dataset (dataset tuple): The dataset involving seed images.
        anchorpoint_dataset (dataset tuple): The dataset involving anchorpoint images.
        learning_rate (float, optional): Learning rate of the optimizer. Defaults to 0.001.
        batch_size (int, optional): Batch size for the poison synthesization. Defaults to 16.
        iters (int, optional): The iteration times of the optimization. Defaults to 1000.
        if_selection (bool, optional): Whether the archorpoint embeedings are selected. 
                                       Defaults to True.
    """

    # Get the ref embeedings
    anchorpoint_embeedings = encoder.predict(anchorpoint_dataset[0]) 
    seed_embeedings = encoder.predict(seed_dataset[0])
    seed_embeedings, sorted_seed_idx = \
                sort_best_match_embeeding_heuristis(anchorpoint_embeedings=anchorpoint_embeedings,
                                                    seed_embeedings=seed_embeedings)
    seed_dataset = (seed_dataset[0][sorted_seed_idx],
                    seed_dataset[1][sorted_seed_idx])
    
    entire_poisons = None
    entire_poison_label = None
    anchorpoint_idx = None
    seed_amount = len(seed_dataset[0])
    
    batch_start_idx = 0
    batch_i = 0

    while batch_start_idx < seed_amount:
        batch_i += 1
        batch_end_idx = np.min([batch_start_idx + batch_size, seed_amount])
        seed = seed_dataset[0][batch_start_idx:batch_end_idx]
        poison_label = seed_dataset[1][batch_start_idx:batch_end_idx]
        
        # Calculate the embeedings of the seed
        seed_embeedings = encoder.predict(seed)

        if if_selection:
            print("w selection")
            selected_anchorpoint_embeedings, selected_anchorpoint_idx = find_nearest_embeedings(seed_embeedings,
                                                                                                anchorpoint_embeedings)
        else:
            print("w/o selection")
            selected_anchorpoint_idx = np.arange(batch_start_idx, batch_end_idx) % len(anchorpoint_embeedings) 
            selected_anchorpoint_embeedings = anchorpoint_embeedings[selected_anchorpoint_idx]
            batch_start_idx += batch_size
            batch_end_idx = np.min([batch_start_idx + batch_size, len(seed_dataset)])

        if anchorpoint_idx is None:
            anchorpoint_idx = selected_anchorpoint_idx
        else:
            anchorpoint_idx = np.r_[anchorpoint_idx, selected_anchorpoint_idx]
            print(anchorpoint_idx.shape)
        
        print('='*20)
        print("Batch: {}".format(batch_i))
        print(seed.shape)
        poisons = craft_poisons_batch(seed, 
                                      selected_anchorpoint_embeedings, 
                                      encoder, 
                                      iters,
                                      learning_rate,
                                      image_scale=image_scale)
        del seed, seed_embeedings, selected_anchorpoint_embeedings 
        if entire_poisons is None:
            entire_poisons = poisons
            entire_poison_label = poison_label
        else:
            entire_poisons = np.r_[entire_poisons, poisons]
            entire_poison_label = np.r_[entire_poison_label, poison_label]

    """
    selected_anchorpoint_dataset = (anchorpoint_dataset[0][anchorpoint_idx],
                                    anchorpoint_dataset[1][anchorpoint_idx])
    """

    #return ((entire_poisons, entire_poison_label), selected_anchorpoint_dataset, anchorpoint_idx) 
    return ((entire_poisons, entire_poison_label), anchorpoint_dataset, sorted_seed_idx) 

def craft_poisons_batch(seed, 
                        anchorpoint_embeedings,
                        encoder,
                        iters,
                        learning_rate=0.01,
                        image_scale=np.array([[-1., 1.] for i in range(3)]).transpose()):

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #opt2 = tf.keras.optimizers.SGD(learning_rate=learning_rate*0.1)
    SMALL_EPS = 1-1e-6
    print(np.min(seed))
    print(np.max(seed))
    # scale seed to [-1,1]
    seed_scaled = 2*(seed-image_scale[0])/(image_scale[1]-image_scale[0])-1
    seed_sacled = seed_scaled.astype('float32')
    w = tf.Variable(np.arctanh(seed_scaled*SMALL_EPS), trainable=True)
    del seed_scaled
    gc.collect()

    BETA0 = 1 
    dim_b = np.cumprod(seed.shape[1:])[-1]
    dim_embeeding = anchorpoint_embeedings.shape[-1]
    beta = BETA0 * (dim_embeeding)**2 / (dim_b)**2 / np.prod((image_scale[0,1]-image_scale[0,0])/2)**2
    print("Beta: {}".format(beta))
    
    init_lr = learning_rate

    decay_step = 50
    start_loss = 1e6
    opt.lr.assign(learning_rate)
    for i in range(iters):
        with tf.GradientTape() as tape:
            #poisons = tf.tanh(w)
            poisons = tf.cast(
                (tf.tanh(w)+1)/2*(image_scale[1]-image_scale[0]) + image_scale[0], tf.float32)
            loss1 = tf.cast(l2(encoder(poisons), anchorpoint_embeedings), tf.float32)
            loss2 = tf.cast(l2(poisons, seed), tf.float32)
            loss = loss1 + beta*loss2
            
        print("Iters:{}, loss:{:.8f}, semantic loss:{:.8f}, visual loss:{:.8f}"
                .format(i+1,
                        loss.numpy()[0],
                        loss1.numpy()[0],
                        loss2.numpy()[0]), end='\r')
        if i % decay_step == 1:
            start_loss = loss.numpy()[0]
        if i % decay_step == 0:
            current_loss = loss.numpy()[0] 
            if current_loss > start_loss * 0.9:
                learning_rate = init_lr * 0.8
                opt.lr.assign(learning_rate)
        gradients = tape.gradient(loss, [w])
        opt.apply_gradients(zip(gradients, [w]))

    poisons = (np.tanh(w.numpy())+1)/2*(image_scale[1]-image_scale[0]) + image_scale[0]

    return poisons

def l2(x, y):
    return tf.reduce_sum(tf.square(x - y), list(range(1, len(x.shape))))