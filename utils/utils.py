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
import numpy as np


def normalize_pose(p3d, p3d_mean, p3d_std):
    # Subtract position of root joint (Hip)
    root = tf.tile(tf.expand_dims(p3d[:, 0, :], axis=1), [1, 17, 1])
    p3d = p3d - root

    p3d = (p3d-p3d_mean) / p3d_std
    return p3d


def normalize_pose_2d(p2d, p2d_mean, p2d_std):
    p2d = (p2d - p2d_mean) / p2d_std
    return p2d


def unnormalize_pose(p3d, p3d_mean, p3d_std):

    b = tf.shape(p3d)[0]

    p3d_17x3 = tf.reshape(p3d, [-1, 17, 3])
    root = p3d_17x3[:, 0, :]
    root = tf.expand_dims(root, axis=1)
    root = tf.tile(root, [1, 17, 1])
    p3d_17x3 = p3d_17x3 - root
    p3d_17x3 = p3d_17x3 * p3d_std[:b, ...] + p3d_mean[:b, ...]
    p3d = tf.reshape(p3d_17x3, [-1, 51])
    return p3d


def unnormalize_pose_2d(p2d, p2d_mean, p2d_std):
    return (p2d * p2d_std) + p2d_mean


def generate_submission(predictions, out_path, two_d=False):
    ids = np.arange(1, predictions.shape[0] + 1).reshape([-1, 1])
    if two_d:
        predictions = predictions.reshape([-1, 34])

    predictions = np.hstack([ids, predictions])
    joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head',
              'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    header = ["Id"]

    for j in joints:
        header.append(j + "_x")
        header.append(j + "_y")
        if not two_d:
            header.append(j + "_z")

    header = ",".join(header)
    np.savetxt(out_path, predictions, delimiter=',', header=header, comments='')


def compute_MPJPE(p3d_out, p3d_gt, p3d_std):

    p3d_out_17x3 = tf.reshape(p3d_out, [-1, 17, 3])
    p3d_gt_17x3 = tf.reshape(p3d_gt, [-1, 17, 3])

    mse = ((p3d_out_17x3 - p3d_gt_17x3) * p3d_std) ** 2
    mse = tf.reduce_sum(mse, axis=2)
    mpjpe = tf.reduce_mean(tf.sqrt(mse))

    return mpjpe


submission_files = [
    "utils.py",
    "resnet_model.py",
    "test.py",
    "training.py",
    "utils.py",
    "vis.py"
]
