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
import tensorflow_probability as tfp
import h5py
import os
import logging
import numpy as np
import pandas as pd
from math import pi
from configs.dataset_config import DatasetConfig


IMAGE_SHAPE = tf.constant([256, 256], dtype=tf.float32)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image,tf.float32) / 128. - 1
    return image


def preprocess_pose(pose):
    pose = tf.cast(pose, tf.float32)
    min_val = 0
    max_val = 255
    normalized_range = max_val - min_val

    image_size = np.array([normalized_range, normalized_range])
    heatmap_size = np.array([64, 64])
    height = 64
    width = 64
    num_joints = 17
    sigma = 0.5
    hm = []
    for i in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = tf.cast((pose[i][0] - min_val) / feat_stride[0], tf.float32)
        mu_y = tf.cast((pose[i][1] - min_val) / feat_stride[1], tf.float32)
        mu_x = tf.floor(mu_x)
        mu_y = tf.floor(mu_y)
        tmp_size = tf.cast(sigma * 3, tf.float32)
        X, Y = tf.meshgrid(np.arange(0, height), np.arange(0, width))
        pos = tf.concat([tf.reshape(X, [-1, 1]), tf.reshape(Y, [-1, 1])], axis=1)
        pos = tf.cast(pos, dtype=tf.float32)
        rv = tfp.distributions.MultivariateNormalFullCovariance(loc=[mu_x, mu_y],
                                                                covariance_matrix=[[tmp_size, 0], [0, tmp_size]])
        hm.append(tf.reshape(300.0 * rv.prob(pos), tf.shape(X))) #TODO: might need a transpose
    hm = tf.stack(hm, axis=2)
    pose = tf.cast(hm, tf.float32)
    return pose


def load_and_preprocess_image_and_pose(path,pose):
    image = tf.read_file(path)
    image = preprocess_image(image)
    image = tf.expand_dims(image, 0)
    pose = tf.expand_dims(pose, 0)
    pose_2d_mean = np.load(DatasetConfig.POSE_2D_MEAN_PATH + '.npy')
    pose_2d_std = np.load(DatasetConfig.POSE_2D_STD_PATH + '.npy')

    pose = tf.cast(pose, tf.float32)
    p2d_mean = tf.cast(tf.constant(pose_2d_mean.reshape([1, 17, 2])), tf.float32)

    p2d_std = tf.cast(tf.constant(pose_2d_std.reshape([1, 17, 2])), tf.float32)

    image, pose, pose_2d_mean, pose_2d_std = homogeneous_augmentation(image, pose, p2d_mean, p2d_std, 1)
    image = image_color_augmentation(image)

    pose = pose[0]
    image = image[0]
    pose = preprocess_pose(pose)
    return image, pose


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    image = preprocess_image(image)
    return image


def create_dataloader_train_2d(data_root, batch_size):
    phase = "train"
    all_image_paths = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines()
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths]
    logging.info("Number of Images creating a loader for {}".format(len(all_image_paths)))
    annotations_path = os.path.join(data_root,"annot","%s.h5"%phase)
    annotations = h5py.File(annotations_path, 'r')
    image_pose_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, annotations['pose2d']))
    image_pose_ds = image_pose_ds.shuffle(buffer_size=len(all_image_paths))
    image_pose_ds = image_pose_ds.map(load_and_preprocess_image_and_pose)
    image_pose_ds = image_pose_ds.repeat() #why do we repeat here
    image_pose_ds = image_pose_ds.batch(batch_size, drop_remainder=True)
    iterator = image_pose_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()
    return dataloader, all_image_paths, annotations


def create_dataloader_test(data_root, batch_size):
    phase = "valid"
    all_image_paths = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines()
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths]
    logging.info("Number of Images creating a loader for {}".format(len(all_image_paths)))
    image_pose_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_pose_ds = image_pose_ds.map(load_and_preprocess_image)

    image_pose_ds = image_pose_ds.batch(batch_size)

    iterator = image_pose_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()

    return dataloader, all_image_paths


def create_dataloader_test_2d(batch_size, csv_file):
    pose_2d_normalized = pd.read_csv(csv_file).drop(['Id'], axis=1)
    pose_ds = tf.data.Dataset.from_tensor_slices(pose_2d_normalized)
    pose_ds = pose_ds.map(load_and_preprocess_2d_pose)

    pose_ds = pose_ds.batch(batch_size)

    iterator = pose_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()

    return dataloader


def load_and_preprocess_2d_pose(pose_2d):
    pose_2d = tf.convert_to_tensor(pose_2d)
    pose_2d = tf.cast(pose_2d, tf.float32)
    return pose_2d


def rotate_data(im, gt2d, gt2dmean, gt2dstd, angle):

    R_2d = tf.stack([[tf.cos(angle), -tf.sin(angle)], [tf.sin(angle), tf.cos(angle)]])

    if im.shape != tf.shape(0, ).shape:
        im = tf.contrib.image.rotate(im, angle)

    gt2d = gt2d - IMAGE_SHAPE / 2
    gt2dmean = gt2dmean - IMAGE_SHAPE / 2

    gt2d = tf.linalg.matmul(gt2d, R_2d)
    gt2dmean = tf.linalg.matmul(gt2dmean, R_2d)
    gt2dstd = tf.linalg.matmul(gt2dstd, R_2d)

    gt2d = gt2d + IMAGE_SHAPE / 2
    gt2dmean = gt2dmean + IMAGE_SHAPE / 2

    return im, gt2d, gt2dmean, gt2dstd


def homogeneous_augmentation(im, p2d_gt, p2d_mean, p2d_std, batch_s):

    # ### Rotating ### #
    def rotate_cond(img, pose2d, pose2dmean, pose2dstd, rand_v):
        angles = tf.convert_to_tensor([-pi / 2, pi / 2])
        samples = tf.multinomial(tf.log([[0.5, 0.5]]), 1)
        angle = angles[tf.cast(samples[0][0], tf.int32)]
        return tf.cond(tf.squeeze(rand_v) < tf.constant(0.5),
                       lambda: rotate_data(img, pose2d, pose2dmean, pose2dstd, angle),
                       lambda: (img, pose2d, pose2dmean, pose2dstd))

    def custom_map(fn, arrays, batch_size, dtype=tf.float32):
        indices = tf.range(batch_size)
        out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=(dtype, ) * 4)
        return out

    flip_prob = tf.random.uniform([batch_s, 1])
    map_lambda = lambda img, pose2d, pose2dmean, pose2dstd, rand: rotate_cond(
        img, pose2d, pose2dmean, pose2dstd, rand)
    im, p2d_gt, p2d_mean, p2d_std = custom_map(map_lambda, (im, p2d_gt, p2d_mean, p2d_std, flip_prob), batch_s)

    # ### Translation ### #
    # Minimum distance between coordinate and edge of image after translation
    margin = tf.tile(tf.constant([[5.0, 5.0]]), (batch_s, 1))

    im_shape = tf.expand_dims(IMAGE_SHAPE, axis=0)
    max_pixel_coords = tf.reduce_max(p2d_gt, axis=1)
    min_pixel_coords = tf.reduce_min(p2d_gt, axis=1)

    max_translation = im_shape - margin - max_pixel_coords
    min_translation = -(min_pixel_coords - margin)
    translation = tf.random.uniform(max_pixel_coords.shape, min_translation, max_translation)
    im = tf.contrib.image.translate(im, translations=translation)

    # Translate 2D coordinates
    translation = tf.tile(tf.expand_dims(translation, axis=1), (1, p2d_gt.shape[1], 1))
    p2d_gt = p2d_gt + translation

    return im, p2d_gt, p2d_mean, p2d_std


def image_color_augmentation(im):

    def additive_gaussian_noise(img, std):
        noise = tf.random_normal(shape=tf.shape(img), mean=0.0, stddev=std, dtype=tf.float32)
        return img + noise

    q = tf.constant(0.25)

    im = tf.cond(tf.squeeze(tf.random.normal([1])) < q, lambda: tf.image.random_brightness(im, 0.1), lambda: im)
    im = tf.cond(tf.squeeze(tf.random.normal([1])) < q, lambda: tf.image.random_hue(im, 0.2), lambda: im)
    im = tf.cond(tf.squeeze(tf.random.normal([1])) < q, lambda: tf.image.random_contrast(im, 0.6, 1.4), lambda: im)
    im = tf.cond(tf.squeeze(tf.random.normal([1])) < q, lambda: tf.image.random_saturation(im, 0.5, 2), lambda: im)
    im = additive_gaussian_noise(im, tf.random.normal([1], stddev=0.1))

    return im
