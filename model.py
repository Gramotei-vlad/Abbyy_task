import tensorflow as tf

from typing import Dict
from tensorflow.keras import layers
from tensorflow.python.framework.ops import Tensor

EMBEDDING_DIM = 200


class NerModel(tf.keras.Model):
    """
    Экземляр LSTM модели;
    """
    def __init__(self, hidden_size: int, num_classes: int, output_key: str = 'predictions'):
        super().__init__(NerModel)

        self.backbone = layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True))
        self.classifier = layers.Dense(num_classes)
        self.output_key = output_key

    @tf.function(input_signature=[tf.TensorSpec(shape=tf.TensorShape([None, None, EMBEDDING_DIM]), dtype=tf.float32)])
    def __call__(self, x: Tensor) -> Dict[str, Tensor]:

        x = self.backbone(x)
        x = self.classifier(x)
        return {self.output_key: x}
