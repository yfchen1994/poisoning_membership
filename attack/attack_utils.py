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


from scipy import spatial
def get_angular_distance(matrix_a, matrix_b):
    return 1 - spatial.cosine(matrix_a, matrix_b)

def find_nearest_embeedings(seed_embeedings, anchorpoint_embeedings, return_distance=False, k=10):
    """For each seed, find the nearest anchorpoint embeeding.

    Args:
        seed_embeedings (np.array): The embeedings of seed images.
        anchorpoint_embeedings (np.array): The embeedings of anchorpoint images.

    Returns:
        tuple: (corresponding anchorpoint_embeedings, anchorpoint indices)
    """
    distance = get_l2_distance(seed_embeedings, anchorpoint_embeedings)
    #distance = get_angular_distance(seed_embeedings, anchorpoint_embeedings)
    nearest_idx = np.argsort(distance, axis=1)
    print(nearest_idx[:,:k].shape)
    if return_distance:
        distances = np.zeros(distance.shape[0])
        for i in range(len(distances)):
            distances[i] = np.mean(distance[i,nearest_idx[i,:k]])
        return distances
    return (anchorpoint_embeedings[nearest_idx[:,0]], nearest_idx[:,0])

def sort_best_match_embeeding_heuristis(anchorpoint_embeedings, seed_embeedings):
    idx1 = []
    idx2 = []
    row_idx = np.arange(len(anchorpoint_embeedings))
    col_idx = np.arange(len(seed_embeedings))
    distance = get_l2_distance(anchorpoint_embeedings, seed_embeedings)
    for _ in range(len(anchorpoint_embeedings)):
        #row_i = np.argmin(np.min(distance,axis=1))
        #col_j = np.argmin(distance[row_i, :])
        row_i = np.argmax(np.max(distance,axis=1))
        col_j = np.argmin(distance[row_i, :])
        idx1.append(row_idx[row_i])
        idx2.append(col_idx[col_j])
        row_idx = np.delete(row_idx, row_i)
        col_idx = np.delete(col_idx, col_j)
        distance = np.delete(distance, row_i, axis=0)
        distance = np.delete(distance, col_j, axis=1)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    sort_idx = idx2[np.argsort(idx1)]
    seed_embeedings = seed_embeedings[sort_idx]
    return (seed_embeedings, sort_idx)

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

def check_mia(poison_config,
              poison_dataset_config,
              attack_config,
              target_class,
              if_plot_curve=False,):

    from attack.main_attack import PoisonAttack
    attack = PoisonAttack(poison_config,
                          poison_dataset_config,
                          attack_config)

    measure_distance_flag = False 

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

    visualize_flag = False 
    if 'clean_label' in poison_config['attack_type'] and visualize_flag:
        normal_dataset = attack.dataset.get_member_dataset()
        poison_dataset = attack.get_poison_dataset()
        anchorpoint_dataset = attack.get_anchorpoint_dataset()
        poison_x, poison_y = poison_dataset
        anchorpoint_x, anchorpoint_y = anchorpoint_dataset
        poison_x = poison_x[np.where(np.argmax(poison_y,axis=1)!=target_class)]
        anchorpoint_x = anchorpoint_x[np.where(np.argmax(poison_y,axis=1)!=target_class)]
        anchorpoint_y = anchorpoint_y[np.where(np.argmax(poison_y,axis=1)!=target_class)]
        poison_y = poison_y[np.where(np.argmax(poison_y,axis=1)!=target_class)]

        poison_dataset = (poison_x, poison_y)
        anchorpoint_dataset = (anchorpoint_x, anchorpoint_y)

        tsne_results = visualize_features(model=model,
                                          clean_dataset=normal_dataset,
                                          poison_dataset=poison_dataset,
                                          anchorpoint_dataset=anchorpoint_dataset,
                                          target_class=target_class,
                                          encoder_name=attack.target_encoder_name,
                                          attack_type=attack.attack_type,
                                          dataset_name=poison_dataset_config['dataset_name'])

    if 'dirty_label' in poison_config['attack_type'] and visualize_flag:
        normal_dataset = attack.dataset.get_member_dataset()
        poison_dataset = attack.get_poison_dataset()
        real_poison_amount = int((attack.dataset.num_classes-1)/attack.dataset.num_classes*attack.seed_amount)
        tsne_results = visualize_features(model=model,
                                          clean_dataset=normal_dataset,
                                          poison_dataset=(poison_dataset[0][:real_poison_amount],
                                                          poison_dataset[1][:real_poison_amount]),
                                          target_class=target_class,
                                          encoder_name=attack.target_encoder_name,
                                          attack_type=attack.attack_type,
                                          dataset_name=poison_dataset_config['dataset_name'])
    elif poison_config['attack_type'] == 'adversarial_examples' and visualize_flag:
        poison_dataset = attack.get_poison_dataset()
        normal_dataset = attack.dataset.get_member_dataset()
        real_poison_amount = int((attack.dataset.num_classes-1)/attack.dataset.num_classes*attack.seed_amount)
        tsne_results = visualize_features(model=model,
                                          clean_dataset=normal_dataset,
                                          poison_dataset=(poison_dataset[0][:real_poison_amount],
                                                          poison_dataset[1][:real_poison_amount]),
                                          target_class=target_class,
                                          encoder_name=attack.target_encoder_name,
                                          attack_type=attack.attack_type,
                                          dataset_name=poison_dataset_config['dataset_name'])
    if visualize_flag:
        with lzma.open('{}_{}_{}_{}_tsne_results.xz'.format(poison_dataset_config['dataset_name'], 
                                                    attack.target_encoder_name, 
                                                    attack.attack_type,
                                                    target_class), 'wb') as f:
            pickle.dump(tsne_results, f)

    if measure_distance_flag:

        # The old implementation
        member_dataset = attack.dataset.get_member_dataset()
        poison_dataset = attack.get_poison_dataset()
        score_clean, score_poisoned = member_heuristics(model,
                                                        member_dataset,
                                                        target_class=target_class,
                                                        poison_dataset=poison_dataset)
        """
        scores=measure_nearest_distance(model=model,
                           member_dataset=member_dataset,
                           whole_member_dataset=whole_member_dataset,
                           #nonmember_dataset=nonmember_dataset,
                           poison_dataset=poison_dataset,
                           target_class=target_class,
                           encoder_name=attack.target_encoder_name,
                           dataset_name=poison_dataset_config['dataset_name'],
                           attack_type=attack.attack_type)
        """
        record_path = 'scores/{}_{}_{}_{}_scores.xz'.format(poison_dataset_config['dataset_name'], 
                                                      attack.target_encoder_name, 
                                                      attack.attack_type,
                                                      target_class)
        
        check_directory(os.path.dirname(record_path))

        with lzma.open(record_path, 'wb') as f:
            pickle.dump(score_poisoned, f)  
        
        if 'finetuned' in attack.target_encoder_name:
            model = attack.get_clean_model()
            score_clean, score_poisoned = member_heuristics(model,
                                                            member_dataset,
                                                            target_class=target_class,
                                                            poison_dataset=poison_dataset)

        record_path = record_path.replace(attack.attack_type, 'clean_model')
        with lzma.open(record_path, 'wb') as f:
            pickle.dump(score_clean, f)  


    return

    del model
    gc.collect()
    print("Diff:{:.2f}".format(poison_auc-clean_auc))
    print('='*30)

    # Path of the checkpoints
    # "checkpoints/clean_model/target_{target_class}/{model_name}"
    # "checkpoints/{attack_type}/target_{target_class}/{model_name}"
    checkpoint_dir = "./checkpoints/{}/clean_model/{}".format(attack.dataset_name, attack.target_encoder_name)
    member_dataset = attack.dataset.get_member_dataset()
    nonmember_dataset = attack.dataset.get_nonmember_dataset()
    poison_dataset = attack.get_poison_dataset()
    results_path = "./learning_dynamics/{}/clean_model/target_{}/{}".format(attack.dataset_name, target_class, attack.target_encoder_name) 
    results_path = os.path.join(results_path.replace('checkpoints', 'learning_dynamics'), 'learning_dynamics.pkl')
    if os.path.exists(checkpoint_dir):
        print("Inspecting the checkpoints of the clean model")
        model = attack.get_clean_model()
        inspect_checkpoints(model,
                            checkpoint_dir,
                            results_path,
                            member_dataset,
                            nonmember_dataset,
                            poison_dataset,
                            target_class=attack.target_class)

        checkpoint_dir = "./checkpoints/{}/{}/target_{}/{}".format(attack.dataset_name, 
                                                            attack.attack_type,
                                                            attack.target_class,
                                                            attack.target_encoder_name)
        results_path = os.path.join(checkpoint_dir.replace('checkpoints', 'learning_dynamics'), 'learning_dynamics.pkl')
        print("Inspecting the checkpoints of the poisoned model")
        inspect_checkpoints(model,
                            checkpoint_dir,
                            results_path,
                            member_dataset,
                            nonmember_dataset,
                            poison_dataset,
                            target_class=attack.target_class)

def extract_features(model,
                     inputs):
    model.summary()
    feature_extractor = tf.keras.Model(model.inputs,
                                       model.get_layer('feature_extractor').get_output_at(0))
    tf.keras.utils.plot_model(feature_extractor, 'model.png')
    return feature_extractor.predict(inputs).reshape((len(inputs),-1))

def visualize_features(model,
                       clean_dataset,
                       poison_dataset,
                       target_class,
                       encoder_name,
                       dataset_name,
                       attack_type,
                       anchorpoint_dataset=None):
    #clean_features = extract_features(model, clean_dataset[0])
    #poison_features = extract_features(model, poison_dataset[0])
    if anchorpoint_dataset:
        features = extract_features(model,
                                    np.r_[clean_dataset[0], poison_dataset[0], anchorpoint_dataset[0]])
    else:
        features = extract_features(model,
                                    np.r_[clean_dataset[0], poison_dataset[0]])
    #from sklearn.manifold import TSNE
    from MulticoreTSNE import MulticoreTSNE as TSNE

    compressed_features = TSNE(n_jobs=4, n_components=2, perplexity=50, n_iter=5000).fit_transform(features)
    clean_compressed_features = compressed_features[:clean_dataset[0].shape[0],:]
    if anchorpoint_dataset:
        poison_compressed_features = compressed_features[clean_dataset[0].shape[0]:-anchorpoint_dataset[0].shape[0],:]
        anchorpoint_compressed_features = compressed_features[-anchorpoint_dataset[0].shape[0]:,:]
    else:
        poison_compressed_features = compressed_features[clean_dataset[0].shape[0]:,:]

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
    fig, ax = plt.subplots()
    plt.scatter(clean_compressed_features[:,0],
                clean_compressed_features[:,1],
                c=np.argmax(clean_dataset[1],axis=1),
                cmap='Spectral',
                marker='.')
    plt.colorbar(boundaries=np.arange(clean_dataset[1].shape[1]+1)-0.5).set_ticks(np.arange(clean_dataset[1].shape[1]))
    plt.scatter(poison_compressed_features[:,0],
                poison_compressed_features[:,1],
                cmap='Spectral',
                c=np.argmax(poison_dataset[1],axis=1),
                marker='x')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([], [], color='k', marker='.', linestyle='', label='Clean Samples'),
                       Line2D([], [], color='k', marker='x', linestyle='', label='Poisons')]
    if anchorpoint_dataset:
        plt.scatter(anchorpoint_compressed_features[:,0],
                    anchorpoint_compressed_features[:,1],
                    c='k',
                    facecolors='none',
                    marker='+')

        legend_elements.append(
        Line2D([], [], color='k', marker='+', markerfacecolor='none', linestyle='', label='Anchorpoints'))

    ax.legend(handles=legend_elements, loc='best')

    plt.title("{}, {}, Target_class:{}".format(encoder_name,
                                               dataset_name,
                                               str(target_class)),
                                               fontsize=FONTSIZE)
    plt.subplots_adjust(bottom=0.15, top=0.97, left=0.08, right=0.99)

    plt.savefig('{}_{}_{}_{}.png'.format(encoder_name,
                                         dataset_name,
                                         attack_type,
                                         str(target_class)), bbox_inches = 'tight')
    
    tsne_results = {
        'clean_compressed_features': clean_compressed_features,
        'clean_labels': clean_dataset[1],
        'poison_compressed_features': poison_compressed_features,
        'poison_labels': poison_dataset[1],
       
    }
    if anchorpoint_dataset:
        tsne_results['anchorpoint_compressed_features'] = anchorpoint_compressed_features
        tsne_results['anchorpoint_labels'] = anchorpoint_dataset[1]

    return tsne_results

def test_sort_function():
    a1 = np.random.randn(40,20)
    b1 = np.random.randn(50,20)
    b1_sorted, _ = sort_best_match_embeeding_heuristis(a1, b1)
    distance = get_l2_distance(a1,b1)
    sum = 0
    for i in range(len(a1)):
        sum += distance[i,i]
    print(sum)
    print(get_l2_distance(a1,b1).shape)
    print(np.trace(get_l2_distance(a1,b1)))
    print(np.trace(get_l2_distance(a1, b1_sorted)))

def change_img_order(anchorpoint_img_dir, seed_img_dir, seed_label_path):
    anchorpoint_idx = np.load(os.path.join(anchorpoint_img_dir, 'idx.npy'))
    seed_labels = np.load(seed_label_path)
    seed_idx = np.arange(len(anchorpoint_idx))
    seed_imgs = load_img_from_dir(seed_img_dir)
    anchorpoint_imgs = load_img_from_dir(anchorpoint_img_dir)
    seed_idx = seed_idx[np.argsort(anchorpoint_idx)]
    anchorpoint_idx = np.sort(anchorpoint_idx)
    anchorpoint_imgs = anchorpoint_imgs[seed_idx]
    seed_imgs = seed_imgs[seed_idx]
    seed_labels = seed_labels[seed_idx]
    save_img_from_array(seed_imgs, seed_img_dir, rescale=False)
    save_img_from_array(anchorpoint_imgs, anchorpoint_img_dir, rescale=False)
    save_poison_label(seed_labels, seed_label_path)
    os.system('rm ' + os.path.join(anchorpoint_img_dir, 'idx.npy'))
    save_poison_label(seed_idx, os.path.join(seed_img_dir, 'idx.npy'))

def member_heuristics(model,
                      member_dataset,
                      #nonmember_dataset,
                      target_class=0,
                      poison_dataset=None):
    member_x, member_y = member_dataset 
    member_label = np.argmax(member_y, axis=1)
    member_feature = extract_features(model, member_x)
    targeted_member_feature = member_feature[np.where(member_label==target_class)]

    targeted_member_num = len(targeted_member_feature)

    correct_label_feature = targeted_member_feature.copy()
    uncorrect_label_feature = member_feature[np.where(member_label!=target_class)]
    untargeted_member_num = len(uncorrect_label_feature)

    if poison_dataset:
        poison_x, poison_y = poison_dataset 
        poison_label = np.argmax(poison_y, axis=1)
        poison_feature = extract_features(model, poison_x)
        correct_label_feature = np.r_[correct_label_feature,
                                       poison_feature[np.where(poison_label==target_class)]]
        uncorrect_label_feature = np.r_[uncorrect_label_feature,
                                        poison_feature[np.where(poison_label!=target_class)]]
    mc_distances = spatial.distance.cdist(targeted_member_feature, correct_label_feature)    
    mu_distances = spatial.distance.cdist(targeted_member_feature, uncorrect_label_feature)

    mc_distances_clean = np.sort(mc_distances[:,:targeted_member_num], axis=1)
    mu_distances_clean = np.sort(mu_distances[:, :untargeted_member_num], axis=1)
    score_clean = (mu_distances_clean[:,0] - mc_distances_clean[:,1]) / np.min(np.c_[mu_distances_clean[:,0], mu_distances_clean[:,1]], axis=1)

    mc_distances = np.sort(mc_distances, axis=1)
    mu_distances = np.sort(mu_distances, axis=1)
    score_poisoned = (mu_distances[:,0]-mc_distances[:,1]) / np.min(np.c_[mu_distances[:,0], mu_distances[:,1]], axis=1)

    return (score_clean, score_poisoned)

def measure_nearest_distance(model,
                             member_dataset,
                             #nonmember_dataset,
                             whole_member_dataset,
                             poison_dataset,
                             target_class,
                             encoder_name,
                             dataset_name,
                             attack_type,
                             metric='Mentr'):
    memberfeatures = extract_features(model,
                                      member_dataset[0])
    #nonmemberfeatures = extract_features(model,
    #                                     nonmember_dataset[0])
    poison_x, poison_y = poison_dataset
    poison_label = np.argmax(poison_y, axis=1)
    #poison_x = poison_x[np.where(np.argmax(poison_y,axis=1)!=target_class)]
    whole_member_x, whole_member_y = whole_member_dataset
    whole_member_label = np.argmax(whole_member_y, axis=1)
    correct_label_x = np.r_[member_dataset[0],
                            poison_x[np.where(poison_label==target_class)]]
    uncorrect_label_x = np.r_[whole_member_x[np.where(whole_member_label!=target_class)],
                              poison_x[np.where(poison_label!=target_class)]]
    correct_label_features = extract_features(model, correct_label_x)
    uncorrect_label_features = extract_features(model, uncorrect_label_x)


    mc_distances = np.sort(spatial.distance.cdist(memberfeatures, correct_label_features), axis=1)
    mu_distances = np.sort(spatial.distance.cdist(memberfeatures, uncorrect_label_features), axis=1)
    return (mu_distances[:,0]-mc_distances[:,1]) / np.min(np.c_[mu_distances[:,0], mu_distances[:,1]])

    poisonfeatures = extract_features(model,
                                      poison_x)
    
    mm_distances = np.sort(spatial.distance.cdist(memberfeatures, memberfeatures), axis=1)
    mp_distances = np.sort(spatial.distance.cdist(memberfeatures, poisonfeatures), axis=1)
    # The nearest distance to the member
    #scores = np.sum(mp_distances < mm_nearest_distances, axis=1)
    return (mp_distances[:,0] - mm_distances[:,1]) / np.min(np.c_[mp_distances[:,0], mm_distances[:,1]], axis=1)

    #mp_distances = np.sort(spatial.distance.cdist(memberfeatures, poisonfeatures), axis=1)
    #mp_distances = spatial.distance.cdist(memberfeatures, poisonfeatures)
    #member_labels = np.argmax(member_dataset[1], axis=1)
    #pm_target = np.mean(mp_distances[np.where(member_labels==target_class)],axis=0) 
    #pm_nontarget = np.mean(mp_distances[np.where(member_labels!=target_class)],axis=0) 
    #return pm_target / pm_nontarget

    #return np.mean(pm_distances, axis=1) / np.mean(mm_distances[:,1:])
    #return np.mean(mp_distances, axis=1) / np.mean(mm_distances[:, 1:], axis=1)
    
    member_distances = find_nearest_embeedings(memberfeatures, poisonfeatures, return_distance=True,k=10)
    member_distances = member_distances / memberfeatures.shape[1]
    nonmember_distances = find_nearest_embeedings(nonmemberfeatures, poisonfeatures, return_distance=True,k=10)
    nonmember_distances = nonmember_distances / nonmemberfeatures.shape[1]
    member_preds = model.predict(member_dataset[0])
    nonmember_preds = model.predict(nonmember_dataset[0])
    if metric == 'Mentr':
        member_score = -_Mentr(member_preds, member_dataset[1])
        nonmember_score = -_Mentr(nonmember_preds, nonmember_dataset[1])
    elif metric == 'max':
        member_score = _Max(member_preds)
        nonmember_score = _Max(nonmember_preds)
    return (member_distances, nonmember_distances, member_score, nonmember_score)

if __name__ == '__main__':
    #test_sort_function()
    change_img_order(sys.argv[1], sys.argv[2], sys.argv[3])
