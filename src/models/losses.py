import tensorflow as tf


def get_loss():
    return tf.keras.losses.SparseCategoricalCrossentropy()


def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.0003)
