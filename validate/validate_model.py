from configs.testing_config import TestingConfig
from data_loaders.human_pose_ds_loader import HumanPoseDSLoader
from utils.utils import unnormalize_pose, generate_submission
from configs.dataset_config import DatasetConfig
from configs.master_config import CONFIG
import tensorflow as tf
from utils.utils import normalize_pose_2d
from models.pose_hrnet import pose_hrnet_model
from models import twod_threed_regressor
import numpy as np
import math
from tqdm import trange
import os


class Validate(object):
    def __init__(self, two_d_input=False, two_d_pose_file=None, submission_file_name="submission"):
        self.heatmap_size = 64
        self.num_joints = 17
        self.submission_file_name = submission_file_name
        if two_d_input and two_d_pose_file is None:
            raise Exception("Please include the 2d POSE file")
        self.dataset_loader = HumanPoseDSLoader()
        if not two_d_input:
            self.dataset_loader.load_data_test_images()
        else:
            self.dataset_loader.load_data_test_2d_pose(two_d_pose_file)
        return

    def validate_model_public_highresnet_2d_pose(self):
        with tf.Session() as sess:
            model = pose_hrnet_model.Model()
            # predict 2d pose
            p2d_out = model(self.dataset_loader.images_loader, training=False)
            # restore weights
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(TestingConfig.HIGHRES_LOG_PATH, CONFIG.TEST_HIGHRESNET_CHECKPOINT))

            #get maximum of the heatmaps
            values = tf.argmax(tf.reshape(p2d_out, shape=[-1, self.heatmap_size*self.heatmap_size, self.num_joints]), axis=1)
            positions = tf.stack([tf.mod(values, self.heatmap_size), values // self.heatmap_size])
            positions_2d = tf.transpose(positions, perm=[1, 2, 0])

            pose_2d_mean = np.load(DatasetConfig.POSE_2D_MEAN_PATH + '.npy')
            pose_2d_std = np.load(DatasetConfig.POSE_2D_STD_PATH + '.npy')

            p2d_normalized = normalize_pose_2d(positions_2d, pose_2d_mean, pose_2d_std)
            predictions = None
            with trange(math.ceil(10987 / TestingConfig.BATCH_SIZE)) as t:
                for i in t:
                    p2d_out_normalized = sess.run(p2d_normalized)
                    if predictions is None:
                        predictions = p2d_out_normalized
                    else:
                        predictions = np.concatenate([predictions, p2d_out_normalized], axis=0)
        generate_submission(predictions, self.submission_file_name + ".csv", two_d=True)

    def validate_model_public_baseline_3d_regressor(self):
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        with tf.Session(config=config) as sess:
            model = twod_threed_regressor.Model()
            # # predict 3d pose
            p3d_normalized = model(self.dataset_loader.pose_loader, training=False)
            p3d_out = unnormalize_pose(p3d_normalized, self.dataset_loader.p3d_mean, self.dataset_loader.p3d_std)
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(TestingConfig.BASELINE_REGRESSOR_LOG_PATH, CONFIG.TEST_BASELINE_REGRESSOR_CHECKPOINT))

            predictions = None
            with trange(math.ceil(10987 / TestingConfig.BATCH_SIZE)) as t:
                for i in t:
                    p3d_out_value = sess.run(p3d_out)
                    if predictions is None:
                        predictions = p3d_out_value
                    else:
                        predictions = np.concatenate([predictions, p3d_out_value], axis=0)
        generate_submission(predictions,  self.submission_file_name + ".csv.gz")
