from data_loaders.utils import create_dataloader_test, create_dataloader_train_2d, create_dataloader_test_2d
import numpy as np
import os
import tensorflow as tf
from configs.dataset_config import DatasetConfig


class HumanPoseDSLoader(object):
    def __init__(self):
        self.DATA_PATH = DatasetConfig.DATA_PATH
        self.TRAIN_BATCH_SIZE = DatasetConfig.TRAINBATCH_SIZE
        self.TEST_BATCH_SIZE = DatasetConfig.TESTBATCH_SIZE
        self.images_loader = None
        self.p3d_gt_loader = None
        self.p2d_gt_loader = None
        self.p3d_gt = None
        self.p2d_gt = None
        self.p3d_std = None
        self.p3d_mean = None
        self.image_paths = None
        self.num_joints = 17
        self.heatmap_size = np.array([64, 64])
        self.image_size = np.array([256, 256])
        self.use_different_joints_weight = True
        self.joints_weight = 1
        self.sigma = 2
        self.pose_loader = None

    def load_data_train_2d_heatmaps(self):
        data_loader, image_paths, annotations = create_dataloader_train_2d(data_root=self.DATA_PATH,
                                                                           batch_size=self.TRAIN_BATCH_SIZE)
        im_loader, p2d_gt_loader = data_loader
        self.images_loader = im_loader
        self.p2d_gt_loader = p2d_gt_loader
        self.image_paths = image_paths
        self.p3d_gt = annotations["pose3d"]
        self.p2d_gt = annotations["pose2d"]

    def load_data_test_images(self):
        data_loader, image_paths = create_dataloader_test(data_root=self.DATA_PATH, batch_size=self.TEST_BATCH_SIZE)
        # load mean and std
        im_loader = data_loader
        p3d_mean = np.loadtxt(os.path.join(self.DATA_PATH, 'annot', "mean.txt")).reshape([1, 17, 3]).astype(np.float32)
        p3d_std = np.loadtxt(os.path.join(self.DATA_PATH, 'annot', "std.txt")).reshape([1, 17, 3]).astype(np.float32)
        p3d_std = tf.constant(p3d_std)
        p3d_mean = tf.constant(p3d_mean)
        p3d_std = tf.tile(p3d_std, [self.TEST_BATCH_SIZE, 1, 1])
        p3d_mean = tf.tile(p3d_mean, [self.TEST_BATCH_SIZE, 1, 1])
        # normalize 3d pose
        self.images_loader = im_loader
        self.image_paths = image_paths
        self.p3d_std = p3d_std
        self.p3d_mean = p3d_mean

    def load_data_test_2d_pose(self, csv_file):
        data_loader = create_dataloader_test_2d(batch_size=self.TEST_BATCH_SIZE, csv_file=csv_file)
        # load mean and std
        pose_loader = data_loader
        p3d_mean = np.loadtxt(os.path.join(self.DATA_PATH, 'annot', "mean.txt")).reshape([1, 17, 3]).astype(np.float32)
        p3d_std = np.loadtxt(os.path.join(self.DATA_PATH, 'annot', "std.txt")).reshape([1, 17, 3]).astype(np.float32)
        p3d_std = tf.constant(p3d_std)
        p3d_mean = tf.constant(p3d_mean)
        p3d_std = tf.tile(p3d_std, [self.TEST_BATCH_SIZE, 1, 1])
        p3d_mean = tf.tile(p3d_mean, [self.TEST_BATCH_SIZE, 1, 1])
        # normalize 3d pose
        self.pose_loader = pose_loader
        self.p3d_std = p3d_std
        self.p3d_mean = p3d_mean



