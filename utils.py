# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('..')
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf
import PIL


DATASET_ROOT = '../datasets/'

def check_directory(path):
    """
    Check whether the directory ``path'' exists.
    If not, it will be created.

    Args:
        path (string): The path to check.
    """
    if not os.path.exists(path):
        # All the missing intermediate directories will be created
        Path(path).mkdir(parents=True)

def merge_dataset(dataset1, dataset2):
    return (np.r_[dataset1[0], dataset2[0]],
            np.r_[dataset1[1], dataset2[1]])

def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

if __name__ == '__main__':
    pass
