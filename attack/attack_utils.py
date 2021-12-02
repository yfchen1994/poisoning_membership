import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('..')

import gc
import numpy as np
from sklearn.metrics import roc_auc_score 
import tensorflow as tf

import PIL

def load_dataset(img_dir, label_path, img_num):
    return (load_img_from_dir(img_dir, img_num),
            load_poison_label(label_path, img_num))

def load_img_from_dir(img_dir,
                      img_amount):
    imgs = []
    if type(img_amount) is list:
        start_idx, end_idx = img_amount
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
                        img_dir):
    img_array = (img_array + 1) / 2 * 255
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


def find_nearest_embeedings(seed_embeedings, anchorpoint_embeedings):
    """For each seed, find the nearest anchorpoint embeeding.

    Args:
        seed_embeedings (np.array): The embeedings of seed images.
        anchorpoint_embeedings (np.array): The embeedings of anchorpoint images.

    Returns:
        tuple: (corresponding anchorpoint_embeedings, anchorpoint indices)
    """
    distance = [[np.linalg.norm(v1 - v2) for v2 in anchorpoint_embeedings]
                    for v1 in seed_embeedings]
    nearest_idx = np.argmin(distance, axis=1)
    return (anchorpoint_embeedings[nearest_idx], nearest_idx)

def _Mentr(preds, y):
    fy  = np.sum(preds*y, axis=1)
    fi = preds*(1-y)
    score = -(1-fy)*np.log(fy)-np.sum(fi*np.log(1-fi+1e-30), axis=1)
    return score

def _Max(preds):
    return np.max(preds, axis=1)

def mia(model, member_dataset, nonmember_dataset, metric='Mentr'):
    member_preds = model.predict(member_dataset[0])
    nonmember_preds = model.predict(nonmember_dataset[0])
    if metric == 'Mentr':
        member_score = _Mentr(member_preds, member_dataset[1])
        nonmember_score = _Mentr(nonmember_preds, nonmember_dataset[1])
    elif metric == 'max':
        member_score = _Max(member_preds)
        nonmember_score = _Max(nonmember_preds)
    member_label = -np.ones(len(member_preds))
    nonmember_label = np.ones(len(nonmember_preds))
    mia_auc = roc_auc_score(np.r_[member_label, nonmember_label],
                            np.r_[member_score, nonmember_score])
    
    
    member_amount = len(member_label)
    nonmember_amount = len(nonmember_label)
    print("/"*20)
    print('Membership Inference Attack:')
    print('{} Members vs. {} Non-members:'.format(member_amount, nonmember_amount))
    print('MIA AUC:{:.6f}'.format(mia_auc))
    print("/"*20)
    return mia_auc

def evaluate_model(model, testing_dataset):
    # Get the testing accuracy 
    loss, acc = model.evaluate(testing_dataset[0],
                               testing_dataset[1],
                               batch_size=100,
                               verbose=0)
    return acc

def check_mia(poison_config,
              poison_dataset_config,
              attack_config,
              target_class):

    from attack.main_attack import PoisonAttack
    attack = PoisonAttack(poison_config,
                          poison_dataset_config,
                          attack_config)
    """
    member_dataset = attack.dataset.get_member_dataset(target_class=target_class)
    nonmember_dataset = attack.dataset.get_nonmember_dataset(target_class=target_class)
    testing_dataset = attack.dataset.get_nonmember_dataset()
    """
    print('='*30)
    print("Target encoder:{}".format(attack.target_encoder_name))
    print("Poison encoder:{}".format(attack.poison_encoder_name))
    """
    print("Working on clean model.")
    model = attack.get_clean_model()
    clean_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr')
    clean_test_acc_sub = evaluate_model(model, nonmember_dataset)
    clean_test_acc = evaluate_model(model, testing_dataset)
    print("Testing accuracy of the target class: {:.2f}% ({})"\
          .format(clean_test_acc_sub*100, len(nonmember_dataset[0])))
    print("Testing accuracy of the whole dataset: {:.2f}% ({})"\
          .format(clean_test_acc*100, len(testing_dataset[0])))
    del model
    gc.collect()
    print("*"*20)
    """
    print("Poisoning model.")
    model = attack.get_poisoned_model()
    """
    poison_auc = mia(model, member_dataset, nonmember_dataset, metric='Mentr')
    poisoned_test_acc_sub = evaluate_model(model, nonmember_dataset)
    poisoned_test_acc = evaluate_model(model, testing_dataset)
    print("Testing accuracy of the target class: {:.2f}% ({})"\
          .format(poisoned_test_acc_sub*100, len(nonmember_dataset[0])))
    print("Testing accuracy of the whole dataset: {:.2f}% ({})"\
          .format(poisoned_test_acc*100, len(testing_dataset[0])))

    anchorpoint_dataset = attack.get_anchorpoint_dataset()
    print(anchorpoint_dataset[0].shape)
    print(anchorpoint_dataset[1].shape)
    poison_dataset = attack.get_poison_dataset()
    normal_dataset = attack.dataset.get_member_dataset(data_amount=1000)
    visualize_features(model,
                       normal_dataset,
#                       (member_dataset[0][:1000],
#                        member_dataset[1][:1000]),
                       (poison_dataset[0][:100],
                        poison_dataset[1][:100]),
                       anchorpoint_dataset,
                       target_class,
                       attack.target_encoder_name,
                       poison_dataset_config['dataset_name'])
    """
    del model
    gc.collect()
    #print("Diff:{:.2f}".format(poison_auc-clean_auc))
    print('='*30)

def extract_features(model,
                     inputs):
    feature_extractor = tf.keras.Model(model.inputs,
                                       model.get_layer('feature_extractor').output)
    return feature_extractor.predict(inputs).reshape((len(inputs),-1))

def visualize_features(model,
                       clean_dataset,
                       poison_dataset,
                       anchorpoint_dataset,
                       target_class,
                       encoder_name,
                       dataset_name):
    #clean_features = extract_features(model, clean_dataset[0])
    #poison_features = extract_features(model, poison_dataset[0])
    features = extract_features(model,
                                np.r_[clean_dataset[0], poison_dataset[0], anchorpoint_dataset[0]])
    from sklearn.manifold import TSNE
    compressed_features = TSNE(n_components=2).fit_transform(features)
    clean_compressed_features = compressed_features[:clean_dataset[0].shape[0],:]
    poison_compressed_features = compressed_features[clean_dataset[0].shape[0]:-anchorpoint_dataset[0].shape[0],:]
    anchorpoint_compressed_features = compressed_features[-anchorpoint_dataset[0].shape[0]:,:]

    ##################
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as mtick

    FONTSIZE = 30
    legend_FONTSIZE = 25
    medium_legend_FONTSIZE = 20
    small_legend_FONTSIZE = 10
    LINEWIDTH = 2
    MARKERSIZE = 10

    sns.set()
    ##################
    plt.figure()
    plt.scatter(clean_compressed_features[:,0],
                clean_compressed_features[:,1],
                c=np.argmax(clean_dataset[1],axis=1),
                cmap='Spectral',
                marker='.')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.scatter(poison_compressed_features[:,0],
                poison_compressed_features[:,1],
                cmap='Spectral',
                c=np.argmax(poison_dataset[1],axis=1),
                marker='x')
    plt.scatter(anchorpoint_compressed_features[:,0],
                anchorpoint_compressed_features[:,1],
                cmap='Spectral',
                c=np.argmax(anchorpoint_dataset[1],axis=1),
                marker='^')

    plt.title("{}, {}, Target_class:{}".format(encoder_name,
                                               dataset_name,
                                               str(target_class)),
              fontsize=FONTSIZE)
    plt.subplots_adjust(bottom=0.15, top=0.97, left=0.08, right=0.99)

    plt.savefig('{}_{}_{}.png'.format(encoder_name,
                                      dataset_name,
                                      str(target_class)), bbox_inches = 'tight')