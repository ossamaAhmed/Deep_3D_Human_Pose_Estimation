"""Copyright (c) 2019 AIT Lab, ETH Zurich, Xu Chen

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

import tensorflow as tf
import pandas as pd
from math import pi
import numpy as np
import h5py
import os

IMAGE_SHAPE = tf.constant([256, 256], dtype=tf.float32)


def load_and_preprocess_pose(im, pose3d, pose2d):
    pose3d = tf.cast(pose3d, tf.float32)
    pose2d = tf.cast(pose2d, tf.float32)
    return im, pose2d, pose3d


def compute_2d_mean_and_std(data_root):
    phase = "train"
    annotations_path = os.path.join(data_root, "annot", "%s.h5" % phase)
    annotations = h5py.File(annotations_path, 'r')

    return np.mean(annotations["pose2d"], axis=0), np.std(annotations["pose2d"], axis=0)


def create_dataloader_train(data_root, batch_size):
    phase = "train"

    all_image_paths = open(os.path.join(data_root, "annot", "%s_images.txt" % phase)).readlines()
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths]

    annotations_path = os.path.join(data_root, "annot", "%s.h5" % phase)
    annotations = h5py.File(annotations_path, 'r')

    image_pose_ds = tf.data.Dataset.from_tensor_slices(
        (all_image_paths, annotations["pose3d"], annotations["pose2d"]))

    image_pose_ds = image_pose_ds.shuffle(buffer_size=len(all_image_paths))
    image_pose_ds = image_pose_ds.map(load_and_preprocess_pose)
    image_pose_ds = image_pose_ds.repeat()
    image_pose_ds = image_pose_ds.batch(batch_size, drop_remainder=True)

    iterator = image_pose_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()

    return dataloader


def rotate_data(im, gt2d, gt2dmean, gt2dstd, gt3d, gt3dmean, gt3dstd, angle):

    R_2d = tf.stack([[tf.cos(angle), -tf.sin(angle)], [tf.sin(angle), tf.cos(angle)]])
    R_3d = tf.stack([[tf.cos(angle), -tf.sin(angle), 0], [tf.sin(angle), tf.cos(angle), 0], [0, 0, 1]])

    if im.shape != tf.shape(0, ).shape:
        im = tf.contrib.image.rotate(im, angle)

    gt2d = gt2d - IMAGE_SHAPE / 2
    gt2dmean = gt2dmean - IMAGE_SHAPE / 2

    gt2d = tf.linalg.matmul(gt2d, R_2d)
    gt2dmean = tf.linalg.matmul(gt2dmean, R_2d)
    gt2dstd = tf.linalg.matmul(gt2dstd, R_2d)

    gt3d = tf.linalg.matmul(gt3d, R_3d)
    gt3dmean = tf.linalg.matmul(gt3dmean, R_3d)
    gt3dstd = tf.linalg.matmul(gt3dstd, R_3d)

    gt2d = gt2d + IMAGE_SHAPE / 2
    gt2dmean = gt2dmean + IMAGE_SHAPE / 2

    return im, gt2d, gt2dmean, gt2dstd, gt3d, gt3dmean, gt3dstd


def homogeneous_augmentation(im, p2d_gt, p2d_mean, p2d_std, p3d_gt, p3d_mean, p3d_std, batch_s):

    # ### Rotating ### #
    def rotate_cond(img, pose2d, pose2dmean, pose2dstd, pose3d, pose3dmean, pose3dstd, rand_v):
        angles = tf.convert_to_tensor([-pi, -pi / 2, pi/2])
        samples = tf.multinomial(tf.log([[1/3, 1/3, 1/3]]), 1)
        angle = angles[tf.cast(samples[0][0], tf.int32)]
        return tf.cond(tf.squeeze(rand_v) < tf.constant(0.5),
                       lambda: rotate_data(img, pose2d, pose2dmean, pose2dstd, pose3d, pose3dmean, pose3dstd, angle),
                       lambda: (img, pose2d, pose2dmean, pose2dstd, pose3d, pose3dmean, pose3dstd))

    def custom_map(fn, arrays, batch_size, dtype=tf.float32):
        indices = tf.range(batch_size)
        out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=(dtype, ) * 7)
        return out

    flip_prob = tf.random.uniform([batch_s, 1])
    map_lambda = lambda img, pose2d, pose2dmean, pose2dstd, pose3d, pose3dmean, pose3dstd, rand: rotate_cond(
        img, pose2d, pose2dmean, pose2dstd, pose3d, pose3dmean, pose3dstd, rand)
    im, p2d_gt, p2d_mean, p2d_std, p3d_gt, p3d_mean, p3d_std = \
        custom_map(map_lambda, (im, p2d_gt, p2d_mean, p2d_std, p3d_gt, p3d_mean, p3d_std, flip_prob), batch_s)

    # ### Translation ### #
    # Minimum distance between coordinate and edge of image after translation
    margin = tf.tile(tf.constant([[5.0, 5.0]]), (batch_s, 1))

    im_x_shape = tf.tile(tf.expand_dims(tf.slice(IMAGE_SHAPE, [0], [1]), axis=0), [batch_s, 1])
    im_y_shape = tf.tile(tf.expand_dims(tf.slice(IMAGE_SHAPE, [1], [1]), axis=0), [batch_s, 1])
    im_shape = tf.concat([im_x_shape, im_y_shape], axis=1)

    max_pixel_coords = tf.reduce_max(p2d_gt, axis=1)
    min_pixel_coords = tf.reduce_min(p2d_gt, axis=1)

    max_translation = im_shape - margin - max_pixel_coords
    min_translation = -(min_pixel_coords - margin)
    translation = tf.random.uniform(max_pixel_coords.shape, min_translation, max_translation)
    im = tf.contrib.image.translate(im, translations=translation)

    # Translate 2D coordinates
    translation = tf.tile(tf.expand_dims(translation, axis=1), (1, p2d_gt.shape[1], 1))
    p2d_gt = p2d_gt + translation
    p2d_mean = p2d_mean + translation

    return im, p2d_gt, p2d_mean, p2d_std, p3d_gt, p3d_mean, p3d_std


def create_dataloader_test_2d(batch_size, csv_file):
    pose_2d_normalized = pd.read_csv(csv_file).drop(['Id'], axis=1)
    pose_ds = tf.data.Dataset.from_tensor_slices(pose_2d_normalized)
    pose_ds = pose_ds.map(load_and_preprocess_2d_pose)

    pose_ds = pose_ds.batch(batch_size, drop_remainder=True)

    iterator = pose_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()

    return dataloader


def load_and_preprocess_2d_pose(pose_2d):
    pose_2d = tf.convert_to_tensor(pose_2d)
    pose_2d = tf.cast(pose_2d, tf.float32)
    return pose_2d
