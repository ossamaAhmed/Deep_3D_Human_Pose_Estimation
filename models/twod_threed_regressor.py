import tensorflow as tf


class Model(object):
    def __init__(self):
        self.joints_size = 51

    def __call__(self, inputs, training):
        with tf.name_scope("regressor_model"):
            outputs = tf.layers.dense(inputs, units=1024, activation=None, use_bias=True)
            outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.dropout(outputs, rate=0.5)
            # get the two layers
            outputs = self.block(outputs, training=training)
            outputs = self.block(outputs, training=training)
            # get last layer
            outputs = tf.layers.dense(outputs, units=self.joints_size, activation=None, use_bias=True)
            # TODO: missing clip the norm of the weights
        return outputs

    def block(self, inputs, training):
        with tf.name_scope("block"):
            residual = tf.identity(inputs)
            outputs = tf.layers.dense(inputs, units=1024, activation=None, use_bias=True)
            outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.dropout(outputs, rate=0.5)

            outputs = tf.layers.dense(outputs, units=1024, activation=None, use_bias=True)
            outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.dropout(outputs, rate=0.5)

            outputs += residual
            outputs = tf.nn.relu(outputs)
            outputs = tf.identity(outputs)
            return outputs
