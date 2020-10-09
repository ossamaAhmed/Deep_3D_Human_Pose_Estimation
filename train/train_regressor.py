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
import os
import gc
import time
import math
from tqdm import trange
from utils.utils import compute_MPJPE, normalize_pose, normalize_pose_2d
from data_loaders.regressor_dataloader import create_dataloader_train, homogeneous_augmentation, compute_2d_mean_and_std
from models import twod_threed_regressor
from configs.training_config import TrainingConfig
from configs.master_config import CONFIG as MasterConfig


class TrainRegressor:
    
    def __init__(self):
        self.data_path = MasterConfig.DATAPATH
        self.n_epochs = MasterConfig.TRAIN_EPOCHS
        self.log_path = MasterConfig.TRAIN_LOG_PATH

        self.batch_size = TrainingConfig.BATCH_SIZE
        self.steps_per_epoch = TrainingConfig.NUM_SAMPLES / self.batch_size
        self.checkpoint_path = TrainingConfig.CHECKPOINT_DIR
        self.lr = TrainingConfig.LEARNING_RATE
        self.n_samples = TrainingConfig.NUM_SAMPLES

        return
    
    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        
        with tf.name_scope("data_loading"):
            # load image and GT 3d pose
            _, p2d_gt, p3d_gt = create_dataloader_train(data_root=self.data_path, batch_size=self.batch_size)
        
            # load mean and std
            p3d_mean_np = np.loadtxt(os.path.join(self.data_path, 'annot', "mean.txt")).reshape([1, 17, 3])
            p3d_std_np = np.loadtxt(os.path.join(self.data_path, 'annot', "std.txt")).reshape([1, 17, 3])
            p3d_mean_np = p3d_mean_np.astype(np.float32)
            p3d_std_np = p3d_std_np.astype(np.float32)
        
            p3d_std = tf.constant(p3d_std_np)
            p3d_mean = tf.constant(p3d_mean_np)
        
            p3d_std = tf.tile(p3d_std, [self.batch_size, 1, 1])
            p3d_mean = tf.tile(p3d_mean, [self.batch_size, 1, 1])
        
            p2d_mean_np, p2d_std_np = compute_2d_mean_and_std(self.data_path)
            p2d_std = tf.cast(tf.constant(p2d_std_np.reshape([1, 17, 2])), tf.float32)
            p2d_mean = tf.cast(tf.constant(p2d_mean_np.reshape([1, 17, 2])), tf.float32)
        
            p2d_std = tf.tile(p2d_std, [self.batch_size, 1, 1])
            p2d_mean = tf.tile(p2d_mean, [self.batch_size, 1, 1])
        
            im_fake = tf.constant(0, shape=(self.batch_size, 0))
        
            _, p2d_gt_aug, p2d_mean, p2d_std, p3d_gt_aug, p3d_mean, p3d_std = homogeneous_augmentation(
                im_fake, p2d_gt, p2d_mean, p2d_std, p3d_gt, p3d_mean, p3d_std, self.batch_size)
        
            p3d_gt_norm = normalize_pose(p3d_gt_aug, p3d_mean, p3d_std)
            p2d_gt_norm = normalize_pose_2d(p2d_gt_aug, p2d_mean, p2d_std)
                
        with tf.name_scope("model_prediction"):
        
            p3d_out = twod_threed_regressor.Model()(tf.layers.flatten(p2d_gt_norm), training=True)
        
        with tf.name_scope("loss"):
        
            main_loss = tf.losses.mean_squared_error(tf.layers.flatten(p3d_gt_norm), p3d_out)
        
        with tf.name_scope("metrics"):
        
            mpjpe = compute_MPJPE(p3d_out, p3d_gt_norm, p3d_std)
        
        with tf.name_scope("train_op"):
        
            global_step = tf.train.get_or_create_global_step()
        
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            trainable_params = [var for var in tf.trainable_variables()]
        
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        
            with tf.control_dependencies(update_ops):
                grads_and_vars = optimizer.compute_gradients(main_loss + reg_loss, var_list=trainable_params)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        with tf.name_scope("summaries"):
        
            tf.summary.scalar("mpjpe", mpjpe, collections=["train_step"])
            tf.summary.scalar("loss", main_loss, collections=["train_step"])
        
            for g, v in grads_and_vars:
                tf.summary.histogram(v.name, v, collections=["train_step"])
                tf.summary.histogram(v.name + '_grad', g, collections=["train_step"])
        
            step_summaries = tf.summary.merge(tf.get_collection('train_step'))
        
        with tf.name_scope("param_count"):
            trainable_count = tf.reduce_sum([tf.reduce_prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        
        # define model saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        
        default_graph = tf.get_default_graph()
        sv = tf.Session(graph=default_graph)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        
        with sv as sess:
            writer = tf.summary.FileWriter(self.log_path, sess.graph)
        
            sample_loss = math.inf
            sample_mpjpe = math.inf
        
            start_time = time.time()
            print("Trainable params: {0}".format(sess.run(trainable_count)))
        
            with tf.name_scope("checkpoint_check"):
                ckpt = tf.train.get_checkpoint_state(self.log_path)
                try:
                    if ckpt and ckpt.model_checkpoint_path and MasterConfig.RESUME_TRAINING:
                        print("Resume training from {0}".format(tf.train.latest_checkpoint(self.log_path)))
                        saver.restore(sess, save_path=tf.train.latest_checkpoint(self.log_path))
                    else:
                        print("Starting a new training session")
                        tf.global_variables_initializer().run()
                except:
                    print("Unable to resume training. Re-initializing weights...")
                    tf.global_variables_initializer().run()
        
            # Training loop
            with trange(int(self.n_epochs * self.steps_per_epoch)) as t:
        
                for i in t:
                    # Basic fetches
                    fetches = {
                        "global_step": global_step,
                        "train_op": train_op
                    }
        
                    # Add summary fetches
                    if i % TrainingConfig.LOG_ITER_FREQ == 0:
                        fetches["summaries"] = step_summaries
                        fetches["loss"] = main_loss
                        fetches["mpjpe"] = mpjpe
        
                    results = sess.run(fetches=fetches, options=run_options)
                    step = results["global_step"]
        
                    # Log scalars, gradients and histograms
                    if i % TrainingConfig.LOG_ITER_FREQ == 0:
                        sample_loss = results["loss"]
                        sample_mpjpe = results["mpjpe"]
                        writer.add_summary(results["summaries"], step)
        
                    # display training status
                    epoch_cur = i * self.batch_size / self.n_samples
        
                    t.set_postfix({
                        "time": "{:.3f}".format(time.time() - start_time),
                        "epoch": epoch_cur,
                        "loss": "%.3f" % sample_loss,
                        "mpjpe": "%.3f" % sample_mpjpe
                    })
        
                    # save model
                    if i % TrainingConfig.SAVE_ITER_FREQ == 0 and i > 0:
                        saver.save(sess, self.checkpoint_path, write_meta_graph=False, global_step=i)
        
                    gc.collect()
        
            step = int(self.n_epochs * self.steps_per_epoch)
            saver.save(sess, self.checkpoint_path, write_meta_graph=False, global_step=step)
        
        print("-------------------------------------------")
        print("----- Training completed successfully -----")
        print("-------------------------------------------")
