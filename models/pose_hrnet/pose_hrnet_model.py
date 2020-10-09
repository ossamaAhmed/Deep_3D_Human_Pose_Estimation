import tensorflow as tf
from configs.pose_hrnet_config import POSE_HIGH_RESOLUTION_NET
from models.pose_hrnet import basic_block, bottle_neck, high_resolution_module


class Model(object):
    def __init__(self):
        self.inplanes = 64
        self.image_size = [256, 256, 3]
        self.size_of_joints = 17
        self.is_training = False
        self.blocks_dict = {'BASIC': basic_block,
                            'BOTTLENECK': bottle_neck}
        self.stage2_cfg = POSE_HIGH_RESOLUTION_NET.STAGE2
        num_channels = self.stage2_cfg.NUM_CHANNELS
        block = self.blocks_dict[self.stage2_cfg.BLOCK]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(num_channels_pre_layer=[256],
                                                       num_channels_cur_layer=num_channels)
        self.stage2, prestage_channels = self._make_stage(self.stage2_cfg, num_inchannels=num_channels)
        self.stage3_cfg = POSE_HIGH_RESOLUTION_NET.STAGE3
        num_channels = self.stage3_cfg.NUM_CHANNELS
        block = self.blocks_dict[self.stage3_cfg.BLOCK]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(num_channels_pre_layer=prestage_channels,
                                                       num_channels_cur_layer=num_channels)
        self.stage3, prestage_channels = self._make_stage(self.stage3_cfg, num_inchannels=num_channels)

        self.stage4_cfg = POSE_HIGH_RESOLUTION_NET.STAGE4
        num_channels = self.stage4_cfg.NUM_CHANNELS
        block = self.blocks_dict[self.stage4_cfg.BLOCK]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(num_channels_pre_layer=prestage_channels,
                                                       num_channels_cur_layer=num_channels)
        self.stage4, prestage_channels = self._make_stage(self.stage4_cfg,
                                                          num_inchannels=num_channels,
                                                          multi_scale_output=False)

        self.init_saver()

    def __call__(self, inputs, training):
        with tf.name_scope("PoseHRNet"):
            inputs_identity = tf.identity(inputs)
            #STEM NET
            outputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=3, strides=2,
                                       use_bias=False, padding='SAME')
            outputs = tf.layers.batch_normalization(inputs=outputs, momentum=POSE_HIGH_RESOLUTION_NET.BN_MOMENTUM,
                                                    training=training)
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.conv2d(inputs=outputs, filters=64, kernel_size=3, strides=2,
                                       use_bias=False, padding='SAME')
            outputs = tf.layers.batch_normalization(inputs=outputs, momentum=POSE_HIGH_RESOLUTION_NET.BN_MOMENTUM,
                                                    training=training)
            outputs = tf.nn.relu(outputs)

            layer_1_output = self._get_layer_output("BootleNeck_1", outputs, training=training, block=bottle_neck,
                                                    planes=64, blocks=4)
            outputs = tf.identity(layer_1_output)
            with tf.name_scope("StageTwo"):
                x_list = []
                for i in range(self.stage2_cfg.NUM_BRANCHES):
                    with tf.name_scope("Transition_Layer_" + str(i)):
                        if self.transition1[i] is not None:
                            if isinstance(self.transition1[i], list):
                                x_list.append(self._inference_sequential(self.transition1[i], outputs, training))
                            else:
                                x_list.append(self.transition1[i](outputs, training=training))
                        else:
                            x_list.append(outputs)
                y_list = self._inference_stage(self.stage2, x_list, training=training)

            with tf.name_scope("StageThree"):
                x_list = []
                for i in range(self.stage3_cfg.NUM_BRANCHES):
                    with tf.name_scope("Transition_Layer_" + str(i)):
                        if self.transition2[i] is not None:
                            if isinstance(self.transition2[i], list):
                                x_list.append(self._inference_sequential(self.transition2[i], y_list[-1], training))
                            else:
                                x_list.append(self.transition2[i](y_list[-1], training=training))
                        else:
                            x_list.append(y_list[i])
                y_list = self._inference_stage(self.stage3, x_list, training=training)

            with tf.name_scope("StageFour"):
                x_list = []
                for i in range(self.stage4_cfg.NUM_BRANCHES):
                    with tf.name_scope("Transition_Layer_" + str(i)):
                        if self.transition3[i] is not None:
                            if isinstance(self.transition3[i], list):
                                x_list.append(self._inference_sequential(self.transition3[i], y_list[-1], training))
                            else:
                                x_list.append(self.transition3[i](y_list[-1], training=training))
                        else:
                            x_list.append(y_list[i])
                y_list = self._inference_stage(self.stage4, x_list, training=training)
            with tf.name_scope("FinalLayer"):
                outputs = tf.layers.conv2d(inputs=y_list[0], filters=self.size_of_joints,
                                           kernel_size=POSE_HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL, strides=1,
                                           padding='SAME' if POSE_HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL == 3
                                           else 'VALID')
        return outputs

    def _get_layer_output(self, namescope, inputs, training, block, planes, blocks, stride=1):
        downsample = None
        with tf.name_scope(namescope):
            outputs = tf.identity(inputs)
            if stride != 1 or self.inplanes != planes * block.expansion:
                def downsample(inputs, training):
                    outputs = tf.layers.conv2d(inputs=inputs, filters=planes * block.expansion, kernel_size=1,
                                     strides=stride, use_bias=False, padding='SAME')
                    outputs = tf.layers.batch_normalization(inputs=outputs, momentum=POSE_HIGH_RESOLUTION_NET.BN_MOMENTUM,
                                                            training=training)
                    return outputs
            layer = block.Model(namescope="layer_0", planes=planes, stride=stride, downsample=downsample)
            outputs = layer(outputs, training=training)
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layer = block.Model("layer_" + str(i), planes=planes)
                outputs = layer(outputs, training=training)
        return outputs

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):

        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    channels = num_channels_cur_layer[i]
                    def block_1(inputs, training):
                        outputs = tf.layers.conv2d(inputs=inputs, filters=channels, kernel_size=3,
                                                   strides=1, use_bias=False, padding='SAME')
                        outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
                        outputs = tf.nn.relu(outputs)
                        return outputs
                    transition_layers.append(block_1)
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i-num_branches_pre else inchannels

                    def block_2(inputs, training):
                        outputs = tf.layers.conv2d(inputs=inputs, filters=outchannels, kernel_size=3,
                                                   strides=2, use_bias=False, padding='SAME')
                        outputs = tf.layers.batch_normalization(inputs=outputs, training=training)
                        outputs = tf.nn.relu(outputs)
                        return outputs
                    conv3x3s.append(block_2)
                transition_layers.append(conv3x3s)
        return transition_layers

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config.NUM_MODULES
        num_branches = layer_config.NUM_BRANCHES
        num_blocks = layer_config.NUM_BLOCKS
        num_channels = layer_config.NUM_CHANNELS
        block = self.blocks_dict[layer_config.BLOCK]
        fuse_method = layer_config.FUSE_METHOD

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(high_resolution_module.Model("high_resolution_module_" + str(i),
                                                        num_branches, block, num_blocks, num_inchannels, num_channels,
                                                        fuse_method, reset_multi_scale_output))
        return modules, modules[-1].get_num_inchannels()

    def _inference_sequential(self, stage_modules, inputs, training):
        outputs = tf.identity(inputs)
        for i in range(len(stage_modules)):
            outputs = stage_modules[i](outputs, training)
        return outputs

    def _inference_stage(self, stage_modules, inputs, training):
        outputs = []
        for i in range(len(inputs)):
            outputs.append(tf.identity(inputs[i]))
        for i in range(len(stage_modules)):
            outputs = stage_modules[i](outputs, training)
        return outputs

    def init_saver(self):
        pass
