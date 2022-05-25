import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf
import gc

def clean_label_attack(encoder,
                       target_class,
                       attack_dataset,
                       seed_amount=0,
                       base_amount=0,
                       poison_config=None,
                       image_scale=np.array([[-1., 1.] for i in range(3)]).transpose()):
    """ The clean label attack.

    Args:
        encoder (keras model): The pretrained feature extractors.
        target_class (int): The subclass that the attacker aims to mount poisoning.
        attack_dataset (dataset tuple): The attacker's dataset.
        seed_amount (int, optional): The amount of the seed images. Defaults to 0.
        base_amount (int, optional): The amount of the base images. Defaults to 0.
        poison_config (dict, optional): Parameters used to craft poisons. Defaults to None.

    Returns:
        dataset tuple: The poisoning dataset.
    """

    class_label = np.argmax(attack_dataset[1], axis=1)
    seed_x = attack_dataset[0][np.where(class_label != target_class)]
    seed_y = attack_dataset[1][np.where(class_label != target_class)]

    class_num = attack_dataset[1].shape[1]

    if seed_amount <= 0:
        seed_amount = len(seed_x)
    if seed_amount < len(seed_x):
        sub_seed_amount = int(seed_amount/(class_num-1))
        seed_x = seed_x[:seed_amount]
        seed_y = seed_y[:seed_amount]
        # Make the seed dataset balanced
        start_idx = 0
        end_idx = start_idx + sub_seed_amount
        for i in range(class_num):
            if i == target_class:
                continue

            sub_x = attack_dataset[0][np.where(class_label==i)]
            sub_y = attack_dataset[1][np.where(class_label==i)]

            # The last round
            if start_idx >= (class_num-2)*sub_seed_amount:
                seed_x[start_idx:] = sub_x[:seed_amount-start_idx]
                seed_y[start_idx:] = sub_y[:seed_amount-start_idx]
            else:
                end_idx = start_idx + sub_seed_amount
                seed_x[start_idx:end_idx] = sub_x[:sub_seed_amount]
                seed_y[start_idx:end_idx] = sub_y[:sub_seed_amount]
                start_idx += sub_seed_amount

    seed_dataset = (seed_x, seed_y)
    del seed_x, seed_y

    base_x = attack_dataset[0][np.where(class_label == target_class)]
    base_y = attack_dataset[1][np.where(class_label == target_class)]

    if base_amount <= 0:
        base_amount = len(base_x)
    if base_amount < len(base_x):
        base_x = base_x[:base_amount]
        base_y = base_y[:base_amount]
    base_dataset = (base_x, base_y)
    del base_x, base_y

    poison_dataset, selected_base_dataset = craft_poisons(encoder,
                                                          seed_dataset,
                                                          base_dataset,
                                                          image_scale=image_scale,
                                                          **poison_config)

    return (poison_dataset, selected_base_dataset)

def craft_poisons(encoder,
                  seed_dataset,
                  base_dataset,
                  learning_rate=0.001,
                  batch_size=16,
                  iters=1000,
                  image_scale=np.array([[-1., 1.] for i in range(3)]).transpose()):

    """ Crafting poisons.

    Args:
        encoder (keras model): The pretrained feature extractors.
        seed_dataset (dataset tuple): The dataset involving seed images.
        base_dataset (dataset tuple): The dataset involving base images.
        learning_rate (float, optional): Learning rate of the optimizer. Defaults to 0.001.
        batch_size (int, optional): Batch size for the poison synthesization. Defaults to 16.
        iters (int, optional): The iteration times of the optimization. Defaults to 1000.
        if_selection (bool, optional): Whether the archorpoint embeedings are selected. 
                                       Defaults to True.
    """
    base_embeedings = encoder.predict(base_dataset[0])

    entire_poisons = None
    entire_poison_label = None
    base_idx = None
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

        selected_base_idx = np.arange(batch_start_idx, batch_end_idx) % len(base_embeedings) 
        selected_base_embeedings = base_embeedings[selected_base_idx]
        batch_start_idx += batch_size
        batch_end_idx = np.min([batch_start_idx + batch_size, len(seed_dataset)])

        if base_idx is None:
            base_idx = selected_base_idx
        else:
            base_idx = np.r_[base_idx, selected_base_idx]
        
        print()
        print('='*20)
        print("Batch: {}".format(batch_i))
        poisons = craft_poisons_batch(seed, 
                                      selected_base_embeedings, 
                                      encoder, 
                                      iters,
                                      learning_rate,
                                      image_scale=image_scale)
        del seed, seed_embeedings, selected_base_embeedings 
        if entire_poisons is None:
            entire_poisons = poisons
            entire_poison_label = poison_label
        else:
            entire_poisons = np.r_[entire_poisons, poisons]
            entire_poison_label = np.r_[entire_poison_label, poison_label]

    return ((entire_poisons, entire_poison_label), base_dataset) 

def craft_poisons_batch(seed, 
                        base_embeedings,
                        encoder,
                        iters,
                        learning_rate=0.01,
                        image_scale=np.array([[-1., 1.] for i in range(3)]).transpose()):

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    eps_inf = 0.05
    lower_bound = seed - eps_inf*(image_scale[1]-image_scale[0])
    lower_bound = np.clip(lower_bound, image_scale[0], image_scale[1])
    upper_bound = seed + eps_inf*(image_scale[1]-image_scale[0])
    upper_bound = np.clip(upper_bound, image_scale[0], image_scale[1])

    seed_scaled = 2*(seed-lower_bound)/(upper_bound-lower_bound)-1
    seed_scaled = seed_scaled.astype('float32')
    SMALL_EPS = 1-1e-6
    seed_scaled = np.clip(seed_scaled, -1.*SMALL_EPS, 1.*SMALL_EPS)

    w = tf.Variable(np.arctanh(seed_scaled*SMALL_EPS), trainable=True)
    del seed_scaled
    gc.collect()

    decay_step = 50
    start_loss = 1e9
    opt.lr.assign(learning_rate)
    for i in range(iters):
        with tf.GradientTape() as tape:
            poisons = (tf.tanh(w)+1)*0.5*(upper_bound-lower_bound) + lower_bound
            loss = l2(encoder(poisons), base_embeedings)
            
        print("Iters:{}, loss:{:.8f}".format(i+1, loss.numpy()[0]), end='\r')
        if i % decay_step == 1:
            start_loss = loss.numpy()[0]
        if i % decay_step == 0:
            current_loss = loss.numpy()[0] 
            if current_loss > start_loss * 0.98:
                learning_rate = learning_rate * 0.8
                print()
                print("Learning_rate: {}".format(learning_rate))
                opt.lr.assign(learning_rate)
        gradients = tape.gradient(loss, [w])
        opt.apply_gradients(zip(gradients, [w]))

    poisons = (np.tanh(w.numpy())+1)*0.5*(upper_bound-lower_bound) + lower_bound 

    return poisons

def l2(x, y):
    return tf.reduce_sum(tf.square(x - y), list(range(1, len(x.shape))))