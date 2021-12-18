import tensorflow as tf

from tensorflow.python.framework.ops import Tensor


def sparse_categorical_focal_loss_with_logits(y_true: Tensor, y_pred: Tensor, gamma: int = 2) -> Tensor:
    """
    Реализация focal loss;
    :param y_true: реальные классы;
    :param y_pred: предсказанные классы;
    :param gamma: параметр гамма из формулы focal loss'а
    :return: рассчитанный лосс;
    """
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    probs = tf.nn.softmax(y_pred, axis=-1)
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true.shape.rank)
    return (1 - probs) ** gamma * ce


@tf.keras.utils.register_keras_serializable()
class SparseCategoricalFocalLossWithLogits(tf.keras.losses.Loss):
    def __init__(self, gamma: int = 2):
        super().__init__()
        self.gamma = gamma

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return sparse_categorical_focal_loss_with_logits(y_true, y_pred, self.gamma)