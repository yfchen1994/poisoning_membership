import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf
import sys
import os
import pickle
from utils import check_directory

def select_subgroup(dataset, target_class):
    # Select the samples from the target class. 
    x, y = dataset
    label = np.argmax(y, axis=1)
    return (x[np.where(label==target_class)], y[np.where(label==target_class)])

def filterout_subgroup(dataset, target_class):
    # Preserve the samples not from the target class.
    # This has the opposite effect of select_subgroup()
    x, y = dataset
    label = np.argmax(y, axis=1)
    return (x[np.where(label!=target_class)], y[np.where(label!=target_class)])

def evaluate_model(model, dataset):
    # Currently, some trained models cannot be compiled correctly.
    # So, here we implement the evaluate_model function to help
    # evaluate the accuracy and loss of the model.
    """
    x, y = dataset
    preds = model.predict(x)
    
    # Calculate accuray
    acc = np.sum(np.argmax(y, axis=1)==np.argmax(preds, axis=1)) / len(y)
    # Calculate the categorical loss entropy
    cce = tf.keras.losses.CategoricalCrossentropy()
    cce_loss = cce(y, preds).numpy()
    return (acc, cce_loss)
    """

    loss, acc = model.evaluate(dataset[0],
                               dataset[1],
                               batch_size=100,
                               verbose=0)
    return (acc, loss)

def evaluate_for_one_epoch(model, dataset):
    class_num = dataset[1].shape[1]

    results = {
        'acc':[],
        'loss':[]
        }
     
    for class_no in range(class_num):
        sub_dataset = select_subgroup(dataset, class_no)
        acc, loss = evaluate_model(model, sub_dataset) 
        results['acc'].append(acc)
        results['loss'].append(loss)

    return results

def inspect_checkpoints(model,
                        checkpoint_dir,
                        results_path,
                        member_dataset,
                        nonmember_dataset,
                        poison_dataset,
                        max_epoch=20):
    # Path of the checkpoints
    # "checkpoints/clean_model/target_{target_class}/{model_name}"
    # "checkpoints/{attack_type}/target_{target_class}/{model_name}"
    ckpt = tf.train.Checkpoint(model)

    class_num = member_dataset[1].shape[1]
    target_class = int(checkpoint_dir.split('/')[3].split('_')[1])

    # In our evaluation, we do not consider the normal samples in the poisoning dataset.
    # These smaples are just used to make the poisoning dataset balanced.
    #poison_dataset = filterout_subgroup(poison_dataset, target_class)

    results = {
        'member': [],
        'nonmember': [],
        'poison': []
    }

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    for epoch in range(max_epoch+1):
        print(epoch)
        ckpt_fname = os.path.join(checkpoint_dir, "cp-{}.ckpt".format(epoch))
        print(ckpt_fname)
        ckpt = tf.train.Checkpoint(model)
        ckpt.restore(ckpt_fname).expect_partial()

        results['member'].append(evaluate_for_one_epoch(model, member_dataset))
        results['nonmember'].append(evaluate_for_one_epoch(model, nonmember_dataset))
        results['poison'].append(evaluate_for_one_epoch(model, poison_dataset))

    results_dir = os.path.dirname(results_path)
    check_directory(results_dir)

    with open(results_path, 'wb') as f:
        pickle.dump(results, f)