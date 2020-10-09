import tensorflow as tf
from models.pose_hrnet.resuable_layers import conv3x3
from configs.pose_hrnet_config import POSE_HIGH_RESOLUTION_NET

expansion = 1


class Model(object):
    def __init__(self, namescope, planes, stride=1, downsample=None):
        self.namescope = namescope
        self.planes = planes
        self.stride = stride
        self.downsample = downsample

    def __call__(self, inputs, training):
        with tf.name_scope(self.namescope):
            residual = tf.identity(inputs)
            outputs = conv3x3(inputs, self.planes, self.stride)
            outputs = tf.layers.batch_normalization(inputs=outputs, momentum=POSE_HIGH_RESOLUTION_NET.BN_MOMENTUM,
                                                    training=training)
            outputs = tf.nn.relu(outputs)
            outputs = conv3x3(outputs, self.planes)
            outputs = tf.layers.batch_normalization(inputs=outputs, momentum=POSE_HIGH_RESOLUTION_NET.BN_MOMENTUM,
                                                    training=training)

            if self.downsample is not None:
                residual = self.downsample(inputs, training)
            outputs += residual
            outputs = tf.nn.relu(outputs)
            outputs = tf.identity(outputs, 'basic_block_output')
        return outputs
