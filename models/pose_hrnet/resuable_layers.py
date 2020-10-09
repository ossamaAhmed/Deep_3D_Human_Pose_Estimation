import tensorflow as tf
from configs.pose_hrnet_config import POSE_HIGH_RESOLUTION_NET


def conv3x3(inputs, filters, strides=1):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
      kernel_size = 3
      pad_total = kernel_size - 1
      pad_beg = pad_total // 2
      pad_end = pad_total - pad_beg
      inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_beg, pad_end],
                                               [pad_beg, pad_end], [0, 0]])

    return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer()) #TODO:double check the initialization here