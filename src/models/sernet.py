import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(x, filters, dilation_rate):
    x = layers.Conv1D(
        filters,
        kernel_size=3,
        padding="same",
        dilation_rate=dilation_rate
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    return x


def attention_block(x):
    """
    Simple attention mechanism
    """
    attention = layers.Dense(1, activation="tanh")(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation("softmax")(attention)
    attention = layers.RepeatVector(x.shape[-1])(attention)
    attention = layers.Permute([2, 1])(attention)

    attended = layers.Multiply()([x, attention])
    return layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(attended)


def build_sernet(input_shape=(171, 1), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    # CNN stem
    x = layers.Conv1D(64, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Dilated Conv blocks
    x = conv_block(x, 64, dilation_rate=1)
    x = conv_block(x, 128, dilation_rate=2)
    x = conv_block(x, 256, dilation_rate=4)

    # Attention pooling
    x = attention_block(x)

    # Dense classifier
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model
