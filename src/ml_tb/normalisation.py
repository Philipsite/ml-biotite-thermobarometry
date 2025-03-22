import tensorflow as tf


class MinMaxScaler(tf.keras.layers.Layer):
    def __init__(self, axis, min=None, max=None, invert=False, **kwargs):
        super(MinMaxScaler, self).__init__(**kwargs)
        if min is not None:
            self.min = tf.convert_to_tensor(min)
        if max is not None:
            self.max = tf.convert_to_tensor(max)
        self.axis = axis
        self.invert = invert

    def adapt(self, data):
        data = tf.convert_to_tensor(data)
        self.min = tf.reduce_min(data, axis=self.axis)
        self.max = tf.reduce_max(data, axis=self.axis)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        if self.invert:
            min_float = tf.cast(self.min, tf.float32)
            max_float = tf.cast(self.max, tf.float32)
            return x * (max_float - min_float) + min_float

        else:
            min_float = tf.cast(self.min, tf.float32)
            max_float = tf.cast(self.max, tf.float32)
            return (x - min_float) / (max_float - min_float)
