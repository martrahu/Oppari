import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class CostSensitiveLoss(tf.keras.losses.Loss):

    def __init__(self, penalty_matrix, name="cost_sensitive_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        # IMPORTANT: keep this as pure Python for serialization
        self.penalty_matrix = penalty_matrix

    def call(self, y_true, y_pred):
        penalty_matrix = tf.constant(self.penalty_matrix, dtype=tf.float32 )
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        penalties = tf.matmul(y_true, penalty_matrix)
        penalty_weight = tf.reduce_sum(penalties * y_pred, axis=1)

        return ce * (1.0 + penalty_weight)

    def get_config(self):
        config = super().get_config()
        config.update({
            "penalty_matrix": self.penalty_matrix
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)