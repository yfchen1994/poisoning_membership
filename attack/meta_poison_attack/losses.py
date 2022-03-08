import tensorflow as tf

def train_loss(y_true, y_pred):
    """Calculate the training loss.

    Args:
        y_true: the ground truth.
        y_pred: model predictions.
    """
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_true, y_pred)
    return loss

class AdvLoss(tf.keras.losses.Loss):
    def __init__(self, target_class, **kwargs):
        self.target_class = target_class
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """Calculate the adversarial loss.
        Attack goal:
        1. maximizing the testing loss on the target class
        2. miminizing the testing loss on other classes. 

        Args:
            y_true: the ground truth.
            y_pred: model predictions.
            target_class (int): the target class to poison. 
        """
        cce = tf.keras.losses.CategoricalCrossentropy(tf.keras.losses.Reduction.SUM)
        indices_target = (tf.math.argmax(y_true, axis=1, output_type=tf.int32) == self.target_class)
        loss = cce(y_true[~indices_target], y_pred[~indices_target]) - \
               cce(y_true[indices_target], y_pred[indices_target])
        return loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "target_class": self.target_class}


def test_adv_loss():
    y_true = tf.constant([[0.1,0.2,0.7],
                          [0.7,0.1,0.2]])
    y_pred = tf.constant([[0.2,0.6,0.2],
                          [0.4,0.3,0.3]])
    target_class = 2
    loss = adv_loss(y_true, y_pred, target_class)
    print("Adv loss: {}".format(loss.numpy()))



if __name__ == '__main__':
    test_adv_loss()

