import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('..')

import gc
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
import pickle
import PIL
import lzma
from attack.inspect_checkpoints import inspect_checkpoints
from utils import check_directory

def clear_previous_kernel():
    tf.keras.backend.clear_session()
    gc.collect()
    tf.compat.v1.reset_default_graph()

def summarize_keras_trainable_variables(model, message):
    s = sum(map(lambda x: x.sum(), model.get_weights()))
    print("summary of trainable variables %s: %.13f" % (message, s))
    return s

def load_dataset(img_dir, label_path, img_num):
    return (load_img_from_dir(img_dir, img_num),
            load_poison_label(label_path, img_num))

def load_img_from_dir(img_dir,
                      img_amount=None):
    imgs = []
    if type(img_amount) is list:
        start_idx, end_idx = img_amount
    elif img_amount is None:
        start_idx = 0
        end_idx = len(os.listdir(img_dir))
    else:
        start_idx = 0
        end_idx = img_amount
    for idx in range(start_idx, end_idx):
        img_path = os.path.join(img_dir, "{}.png".format(idx))
        if os.path.exists(img_path):
            img = np.array(PIL.Image.open(img_path))
            imgs.append(img)
        else:
            break
    return np.array(imgs)

def save_img_from_array(img_array,
                        img_dir,
                        preprocess_type='tensorflow',
                        preprocess_mean=None):
    if preprocess_type == 'tensorflow':
        img_array = (img_array + 1) / 2 * 255
    elif preprocess_type == 'caffe':
        for i in range(3):
            img_array[..., i] += preprocess_mean[i]
        # BGR to RGB
        img_array = img_array[..., ::-1]

    img_array = np.clip(img_array, 0, 255)
    img_array = img_array.astype('uint8')
    for i in range(img_array.shape[0]):
        img = PIL.Image.fromarray(img_array[i])
        img_path = os.path.join(img_dir, "{}.png".format(i))
        img.save(img_path)

def save_poison_label(poison_label,
                      poison_label_path):
    np.save(poison_label_path, poison_label)

def load_poison_label(poison_label_path,
                      poison_num):
    if type(poison_num) is list:
        start_idx, end_idx = poison_num
    else:
        start_idx = 0
        end_idx = poison_num
    poison_label_complete = np.load(poison_label_path)
    if end_idx == start_idx < len(poison_label_complete):
        return poison_label_complete
    else:
        return poison_label_complete[start_idx:end_idx]
    
def get_l2_distance(e1, e2):
    distance = [[np.linalg.norm(v1 - v2) for v2 in e2]
                    for v1 in e1]
    return np.array(distance)

def poison_attack(poison_config,
                  poison_dataset_config,
                  attack_config):
    from attack.main_attack import PoisonAttack
    attack = PoisonAttack(poison_config,
                          poison_dataset_config,
                          attack_config)
    model = attack.get_poisoned_model()
    del model,attack
    # Clear the kernel
    # https://github.com/tensorflow/tensorflow/issues/19671
    tf.keras.backend.clear_session()
    gc.collect()

def _Mentr(preds, y):
    fy = np.sum(preds*y, axis=1)
    fi = preds*(1-y)
    score = -(1-fy)*np.log(fy+1e-30)-np.sum(fi*np.log(1-fi+1e-30), axis=1)
    return score

def _Max(preds):
    return np.max(preds, axis=1)

def mia(model, member_dataset, nonmember_dataset, metric='Mentr', if_plot_curve=False):
    member_preds = model.predict(member_dataset[0])
    nonmember_preds = model.predict(nonmember_dataset[0])
    if metric == 'Mentr':
        member_score = -_Mentr(member_preds, member_dataset[1])
        nonmember_score = -_Mentr(nonmember_preds, nonmember_dataset[1])
    elif metric == 'max':
        member_score = _Max(member_preds)
        nonmember_score = _Max(nonmember_preds)
    member_label = np.ones(len(member_preds))
    nonmember_label = np.zeros(len(nonmember_preds))
    mia_auc = roc_auc_score(np.r_[member_label, nonmember_label],
                            np.r_[member_score, nonmember_score])
    if if_plot_curve:
        fpr, tpr, _ = roc_curve(np.r_[member_label, nonmember_label],
                                np.r_[member_score, nonmember_score])
    
    member_amount = len(member_label)
    nonmember_amount = len(nonmember_label)
    print("/"*20)
    print('Membership Inference Attack:')
    print('{} Members vs. {} Non-members:'.format(member_amount, nonmember_amount))
    print('MIA AUC:{:.6f}'.format(mia_auc))
    print("/"*20)
    if if_plot_curve:
        return (mia_auc, fpr, tpr)
    else:
        return mia_auc

def evaluate_model(model, testing_dataset):
    # Get the testing accuracy
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    loss, acc = model.evaluate(testing_dataset[0],
                               testing_dataset[1],
                               batch_size=100,
                               verbose=0)
    return acc

def check_mia(poison_config,
              poison_dataset_config,
              attack_config,
              if_plot_curve=False,):

    from attack.main_attack import PoisonAttack
    attack = PoisonAttack(poison_config,
                          poison_dataset_config,
                          attack_config)

    target_class = poison_config['target_class']

    member_dataset = attack.dataset.get_member_dataset(target_class=target_class)
    nonmember_dataset = attack.dataset.get_nonmember_dataset(target_class=target_class)
    testing_dataset = attack.dataset.get_nonmember_dataset()
    print('='*30)
    print("Target encoder:{}".format(attack.target_encoder_name))
    print("Poison encoder:{}".format(attack.poison_encoder_name))
    print("Working on clean model.")
    model = attack.get_clean_model()
    if if_plot_curve:
        clean_auc, fpr, tpr = mia(model, member_dataset, nonmember_dataset, metric='Mentr', if_plot_curve=True)
        np.save('clean_fpr.npy', fpr)
        np.save('clean_tpr.npy', tpr) 
    else:
        clean_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr', if_plot_curve=False)
    clean_test_acc_sub = evaluate_model(model, nonmember_dataset)
    clean_test_acc = evaluate_model(model, testing_dataset)
    print("Testing accuracy of the target class: {:.2f}% ({})"\
          .format(clean_test_acc_sub*100, len(nonmember_dataset[0])))
    print("Testing accuracy of the whole dataset: {:.2f}% ({})"\
          .format(clean_test_acc*100, len(testing_dataset[0])))

    del model

    gc.collect()
    print("*"*20)
    print("Poisoning model.")
    model = attack.get_poisoned_model()
    if if_plot_curve:
        poison_auc, fpr, tpr = mia(model, member_dataset, nonmember_dataset, metric='Mentr', if_plot_curve=True)
        np.save('poison_fpr.npy', fpr)
        np.save('poison_tpr.npy', tpr)
    else:
        poison_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr', if_plot_curve=False)
    poisoned_test_acc_sub = evaluate_model(model, nonmember_dataset)
    poisoned_test_acc = evaluate_model(model, testing_dataset)
    print("Testing accuracy of the target class: {:.2f}% ({})"\
          .format(poisoned_test_acc_sub*100, len(nonmember_dataset[0])))
    print("Testing accuracy of the whole dataset: {:.2f}% ({})"\
          .format(poisoned_test_acc*100, len(testing_dataset[0])))

    del model
    gc.collect()
    print("MIA AUC Diff:{:.2f}".format(poison_auc-clean_auc))
    print('='*30)

if __name__ == '__main__':
    pass
