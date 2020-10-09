import tensorflow as tf
from models.pose_hrnet.resuable_layers import conv3x3
from configs.pose_hrnet_config import POSE_HIGH_RESOLUTION_NET


class Model(object):
    def __init__(self, namescope, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        self.namescope = namescope
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        self.num_inchannels = num_inchannels
        self.multi_scale_output = multi_scale_output
        self.num_channels = num_channels
        self.blocks = blocks
        self.num_blocks = num_blocks
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

    def __call__(self, inputs, training):
        with tf.name_scope(self.namescope):
            outputs = []
            for i in range(len(inputs)):
                outputs.append(tf.identity(inputs[i]))
            if self.num_branches == 1:
                return [self._call_branch(self.branches[0], inputs[0], training=training)]
            for i in range(self.num_branches):
                outputs[i] = self._call_branch(self.branches[i], inputs[i], training=training)

            x_fuse = []

            for i in range(len(self.fuse_layers)):
                if isinstance(self.fuse_layers[i][0], list):
                    y = outputs[0] if i == 0 else self._inference_sequential(self.fuse_layers[i][0], outputs[0],
                                                                             training=training)
                else:
                    y = outputs[0] if i == 0 else self.fuse_layers[i][0](outputs[0], training=training)
                for j in range(1, self.num_branches):
                    if i == j:
                        y = y + outputs[j]
                    else:
                        if isinstance(self.fuse_layers[i][j], list):
                            y = y + self._inference_sequential(self.fuse_layers[i][j], outputs[j],
                                                               training=training)
                        else:
                            y = y + self.fuse_layers[i][j](outputs[j], training=training)
                x_fuse.append(tf.nn.relu(y))

            return x_fuse

    def _call_branch(self, branch, inputs, training):
        outputs = tf.identity(inputs)
        for i in range(len(branch)):
            outputs = branch[i](outputs, training)
        return outputs

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return branches

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            filters = num_channels[branch_index] * block.expansion
            def downsample(inputs, training):
                outputs = tf.layers.conv2d(inputs=inputs, filters=filters,
                                           kernel_size=1, strides=stride, use_bias=False, padding='SAME')
                outputs = tf.layers.batch_normalization(inputs=outputs, momentum=POSE_HIGH_RESOLUTION_NET.BN_MOMENTUM,
                                                        training=training)
                return outputs
        layers = []
        layers.append(block.Model(namescope="layer_0", planes=num_channels[branch_index],
                                  stride=stride, downsample=downsample))
        self.num_inchannels[branch_index] = self.num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block.Model(namescope="layer_" + str(i), planes=num_channels[branch_index]))
        return layers

    def get_num_inchannels(self):
        return self.num_inchannels

    def _make_block_layer_1(self, channels, j, i):
        def block(inputs, training):
            my_channels = channels
            outputs = tf.layers.conv2d(inputs=inputs, filters=int(my_channels), kernel_size=1,
                                       strides=1, use_bias=False, padding='VALID')
            outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
            new_height = tf.shape(outputs)[1] * (2 ** (j - i))
            new_width = tf.shape(outputs)[2] * (2 ** (j - i))
            outputs = tf.image.resize_nearest_neighbor(outputs, size=[new_height, new_width])
            return outputs
        return block

    def _make_block_layer_2(self, num_outchannels_conv3x3):
        def block(inputs, training):
            outputs = tf.layers.conv2d(inputs=inputs, filters=num_outchannels_conv3x3, kernel_size=3,
                                       strides=2, use_bias=False, padding='SAME')
            outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
            return outputs
        return block

    def _make_block_layer_3(self, num_outchannels_conv3x3):
        def block(inputs, training):
            outputs = tf.layers.conv2d(inputs=inputs, filters=num_outchannels_conv3x3,
                                       kernel_size=3,
                                       strides=2, use_bias=False, padding='SAME')
            outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
            outputs = tf.nn.relu(outputs)
            return outputs
        return block

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    channels = num_inchannels[i]
                    fuse_layer.append(self._make_block_layer_1(channels, j, i))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(self._make_block_layer_2(num_outchannels_conv3x3))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(self._make_block_layer_2(num_outchannels_conv3x3))
                    fuse_layer.append(conv3x3s)
            fuse_layers.append(fuse_layer)

        return fuse_layers

    def _inference_sequential(self, stage_modules, inputs, training):
        outputs = tf.identity(inputs)
        for i in range(len(stage_modules)):
            outputs = stage_modules[i](outputs, training)
        return outputs


