import sys
import os
import argparse
from attack.attack_utils import mia, evaluate_model
parser = argparse.ArgumentParser()
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--device_no', type=str, default='0')


args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_no
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def load_mnist():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    #train_x = 2 * (train_x / 255.) - 1
    #test_x = 2 * (test_x / 255.) - 1
    train_x = train_x / 255.
    test_x = test_x / 255.

    train_x = np.reshape(train_x, (len(train_x), 28, 28, 1))
    test_x = np.reshape(test_x, (len(test_x), 28, 28, 1))

    train_y = tf.keras.utils.to_categorical(train_y, 10)
    test_y = tf.keras.utils.to_categorical(test_y, 10)
    return (train_x, train_y), (test_x, test_y)

def get_encoder(encoder_path, 
                train_x=None,
                train_y=None):
    if os.path.exists(encoder_path):
        return tf.keras.models.load_model(encoder_path)

    input_img = tf.keras.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    encoder = tf.keras.Model(input_img, encoded)
    encoder.compile(optimize='adam', loss='mean_absolute_error')

    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(train_x, train_y,
                    epochs=100,
                    batch_size=100,
                    shuffle=True)
    encoder.save(encoder_path)
    return encoder

def get_classifier(classifier_path, 
                   encoder=None, 
                   train_x=None, 
                   train_y=None):
    if os.path.exists(classifier_path):
        return tf.keras.models.load_model(classifier_path)

    input_img = tf.keras.Input(shape=(28, 28, 1))
    feature = encoder(input_img)
    encoder.trainable = True 
    x = layers.Flatten()(feature)
    x = layers.Dense(128, activation='tanh')(x)
    x = layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(input_img, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(train_x, train_y,
              epochs=100,
              batch_size=100)
    model.save(classifier_path)
    return model

def poison_encoder_retrain(encoder_path, normal_dataset, poisoning_dataset, target_class):
    poisoning_idx = np.where(np.argmax(poisoning_dataset[1],axis=1)==target_class)
    normal_idx = np.where(np.argmax(poisoning_dataset[1],axis=1)!=target_class)
    train_y = poisoning_dataset[0][poisoning_idx]
    train_x = poisoning_dataset[0][normal_idx]
    train_x = train_x[:len(train_y)]
    train_x = np.r_[normal_dataset[0], train_x]
    train_y = np.r_[normal_dataset[0], train_y]
    dirty_encoder = get_encoder(encoder_path=encoder_path,
                                train_x=train_x,
                                train_y=train_y)
    return dirty_encoder 

def poison_encoder_update(clean_encoder, encoder_path, poisoning_dataset, target_class):
    if os.path.exists(encoder_path):
        return tf.keras.models.load_model(encoder_path)

    poisoning_idx = np.where(np.argmax(poisoning_dataset[1],axis=1)==target_class)
    normal_idx = np.where(np.argmax(poisoning_dataset[1],axis=1)!=target_class)
    train_y = clean_encoder.predict(poisoning_dataset[0])
    poison_y = train_y[poisoning_idx]
    train_x = poisoning_dataset[0]
    train_x = train_x[normal_idx]
    train_y = train_y[normal_idx]
    train_y[:len(poison_y)] = poison_y
    clean_encoder.trainable=True
    clean_encoder.compile(optimize='adam', loss='mean_absolute_error')
    clean_encoder.fit(train_x, train_y, epochs=1000, batch_size=100)
    clean_encoder.save(encoder_path)
    return clean_encoder 

def test():
    TARGET_CLASS = args.target_class
    # Split dataset
    (train_x, train_y), (test_x, test_y) = load_mnist()

    clean_encoder_dataset = (train_x[:10000], train_y[:10000])
    dirty_encoder_dataset = (train_x[40000:-10000], train_y[40000:-10000])
    classifier_training_dataset = (train_x[-10000:], train_y[-10000:])
    member_dataset = classifier_training_dataset
    nonmember_dataset = (test_x, test_y)

    # Clean encoder
    clean_encoder_path = './clean_encoder_mnist.h5'
    clean_encoder = get_encoder(clean_encoder_path, 
                                train_x=clean_encoder_dataset[0],
                                train_y=clean_encoder_dataset[0])
    
    clean_classifier_path = './clean_classifier_mnist.h5'
    clean_classifier = get_classifier(clean_classifier_path,
                                      clean_encoder,
                                      train_x=classifier_training_dataset[0],
                                      train_y=classifier_training_dataset[1])
    
    # Dirty encoder
    retrain_flag = False 
    if retrain_flag:
        dirty_encoder_path = './dirty_encoder_mnist_{}.h5'.format(str(TARGET_CLASS))
        dirty_encoder = poison_encoder_retrain(encoder_path=dirty_encoder_path,
                                               normal_dataset=clean_encoder_dataset,
                                               poisoning_dataset=dirty_encoder_dataset,
                                               target_class=TARGET_CLASS)
        dirty_classifier_path = './dirty_classifier_mnist_{}.h5'.format(str(TARGET_CLASS))
    else:
        dirty_encoder_path = './dirty_encoder_updated_mnist_{}.h5'.format(str(TARGET_CLASS))
        dirty_encoder = poison_encoder_update(clean_encoder=clean_encoder,
                                              encoder_path=dirty_encoder_path,
                                              poisoning_dataset=dirty_encoder_dataset,
                                              target_class=TARGET_CLASS)
        dirty_classifier_path = './dirty_classifier_updated_mnist_{}.h5'.format(str(TARGET_CLASS))
    
    # Dirty classifier
    dirty_classifier = get_classifier(dirty_classifier_path,
                                      encoder=dirty_encoder,
                                      train_x=classifier_training_dataset[0],
                                      train_y=classifier_training_dataset[1])

    def select_subset(dataset, target_class):
        idx = np.where(np.argmax(dataset[1],axis=1)==target_class)
        return (dataset[0][idx],dataset[1][idx])

    member_dataset = select_subset(member_dataset, target_class=TARGET_CLASS)
    nonmember_dataset = select_subset(nonmember_dataset, target_class=TARGET_CLASS)
    print("Clean classifier:")
    print("Accuracy : {:.2f}%".format(evaluate_model(clean_classifier, nonmember_dataset)*100))
    mia(clean_classifier, member_dataset, nonmember_dataset)
    
    print("Dirty classifier:")
    print("Accuracy : {:.2f}%".format(evaluate_model(dirty_classifier, nonmember_dataset)*100))
    mia(dirty_classifier, member_dataset, nonmember_dataset)

if __name__ == '__main__':
    test()