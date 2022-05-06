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

class TrainingAccuracyPerClass(tf.keras.metrics.Metric):
    def __init__(self, name='training_accuracy_per_class', **kwargs):
        super(TrainingAccuracyPerClass, self).__init__(name=name, **kwargs)
        self.accuracies = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.accuracies is None:
            self.class_num = y_pred.shape[1]
            self.accuracies = self.add_weight(shape=(self.class_num,),
                                              name='acc_perclass',
                                              initializer='zeros')
            self.per_class_num = tf.Variable(tf.zeros(self.class_num))
            self.correct_class_num = tf.Variable(tf.zeros(self.class_num))

        per_class_num = []
        correct_class_num = []

        label_true = tf.cast(tf.argmax(y_true, axis=-1), dtype=tf.int64)
        label_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.int64)

        for class_no in range(self.class_num):
            correct_class_num.append(tf.math.count_nonzero(tf.math.logical_and(label_true==class_no, label_pred==class_no)))
            per_class_num.append(tf.math.count_nonzero(label_true==class_no))

        self.correct_class_num.assign_add(correct_class_num)
        self.per_class_num.assign_add(per_class_num)

    def reset_states(self):
        self.accuracies.assign(tf.zeros(self.class_num))
        self.per_class_num.assign(tf.zeros(self.class_num))
        self.correct_class_num.assign(tf.zeros(self.class_num))
    
    def result(self):
        self.accuracies.assign(1.*self.correct_class_num/self.per_class_num)
        return self.accuracies

cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
class TrainingLossPerClass(tf.keras.metrics.Metric):
    def __init__(self, name='training_loss_per_class', **kwargs):
        super(TrainingLossPerClass, self).__init__(name=name, **kwargs)
        self.loss_perclass = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.loss_perclass is None:
            self.class_num = y_pred.shape[1]
            self.loss_perclass = self.add_weight(shape=(self.class_num,),
                                          name='loss_perclass',
                                          initializer='zeros')
            self.per_class_num = tf.Variable(tf.zeros(self.class_num))

        per_class_num = []
        per_class_loss = []

        label_true = tf.cast(tf.argmax(y_true, axis=-1), dtype=tf.int64)
        label_preds = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.int64)

        for class_no in range(self.class_num):
            per_class_num.append(tf.math.count_nonzero(label_true==class_no))
            # Select sub data
            y_true_sub = tf.gather(y_true, tf.where(label_true==class_no))
            y_pred_sub = tf.gather(y_pred, tf.where(label_true==class_no))
            per_class_loss.append(cce(y_true_sub, y_pred_sub))

        self.per_class_num.assign_add(per_class_num)
        self.loss_perclass.assign_add(per_class_loss)

    def reset_states(self):
        self.loss_perclass.assign(tf.zeros(self.class_num))
        self.per_class_num.assign(tf.zeros(self.class_num))
    
    def result(self):
        self.loss_perclass.assign(1.*self.loss_perclass/self.per_class_num)
        return self.loss_perclass

if __name__ == '__main__':
    pass