import numpy as np
import tensorflow as tf
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

def adversarial_example_attack(clean_model,
                               attack_dataset,
                               target_class,
                               poison_amount,
                               batch_size=10,
                               image_scale=np.array([[-1., 1.] for i in range(3)]).transpose()):
    adversarial_examples = []
    start_idx = 0

    data, ori_label = attack_dataset
    label = np.argmax(ori_label, axis=1)
    # Select the seeds from the attacker's dataset.
    seed_data = data[np.where(label != target_class)]
    poison_label = ori_label[np.where(label != target_class)]
    
    if poison_amount == 0:
        return ([],[])

    if poison_amount < 0:
        # The poisons include all samples from the target class in the attack dataset.
        poison_amount = len(seed_data)
    if poison_amount >= len(seed_data):
        poison_amount = len(seed_data)

    seed_data = seed_data[:poison_amount]
    poison_label = poison_label[:poison_amount]

    batch_i = 0
    test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()

    while start_idx < poison_amount:
        end_idx = np.min([start_idx+batch_size, poison_amount])
        seed_x = seed_data[start_idx:end_idx]
        target_y = (target_class * np.ones(len(seed_x))).astype('int32')
        batch_i += 1
        pgd_results = projected_gradient_descent(clean_model,
                                                 seed_x,
                                                 eps=0.2,
                                                 eps_iter=0.01,
                                                 nb_iter=50,
                                                 norm=np.inf,
                                                 clip_min=-1.,
                                                 clip_max=1.,
                                                 y=target_y,
                                                 targeted=True)
        pgd_pred = clean_model(pgd_results)
        test_acc_pgd(target_y, pgd_pred)
        print("Batch:{}, success rate:{:.2f}%".format(batch_i, test_acc_pgd.result()*100), end='\r')
        adversarial_examples.append(pgd_results)
        start_idx += batch_size
    adversarial_examples = np.concatenate(adversarial_examples)

    poison_dataset = (adversarial_examples, poison_label)
    return poison_dataset