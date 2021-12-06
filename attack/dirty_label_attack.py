import sys
sys.path.append('..')
import tensorflow as tf
import os

import tensorflow as tf
import numpy as np 
from utils import merge_dataset

def dirty_label_attack(target_class,
                       attack_dataset,
                       poison_amount=0):
    """ In this attack, we just change the label of samples from the target class.
        We don't add any perturbation to the data.
        That's the reason why we call it the 'dirty-label' attack, which is
        different from the 'clean-label' attack.
        This attack can be regarded as a baseline attack in our work.

    Args:
        target_class (int): The subclass that the attacker aims to mount poisoning. 
        attack_dataset (dataset tuple): The attacker's dataset.
        poison_amount (int, optional): The amount of poisons. Defaults to 0.

    Returns:
        dataset tuple: The poisoning dataset.
    """

    data, label = attack_dataset
    label = np.argmax(label, axis=1)
    # Select the poisons from the attacker's dataset.
    poison_data = data[np.where(label == target_class)]
    balancing_data = data[np.where(label != target_class)]
    balancing_label = label[np.where(label != target_class)]

    if poison_amount <= 0:
        # The poisons include all samples from the target class in the attack dataset.
        poison_amount = len(poison_data)
    if poison_amount >= len(poison_data):
        poison_amount = len(poison_data)

    label_range = np.unique(label)
    num_classes = len(label_range)
    label_range = label_range[np.where(label_range != target_class)]
    balancing_amount = int(1./num_classes*poison_amount)
    poison_amount = poison_amount - balancing_amount
    poison_data = poison_data[:poison_amount]

    RANDOM_LABEL = True 
    if RANDOM_LABEL:
        poison_label = np.random.choice(label_range, size=poison_amount, replace=True)
        poison_label = tf.keras.utils.to_categorical(poison_label, num_classes=num_classes)
    else:
        #TODO: We need to design some rules to change the label
        exit(0)
        pass
    balancing_data = balancing_data[:balancing_amount]
    balancing_label = tf.keras.utils.to_categorical(balancing_label[:balancing_amount], num_classes=num_classes)
    poison_dataset = merge_dataset((poison_data, poison_label), 
                                   (balancing_data, balancing_label))

    return poison_dataset