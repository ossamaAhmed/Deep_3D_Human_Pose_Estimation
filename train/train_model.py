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
import os
from tqdm import trange
from models.pose_hrnet import pose_hrnet_model
from configs.training_config import TrainingConfig
from data_loaders.human_pose_ds_loader import HumanPoseDSLoader


class TrainModel(object):
    def __init__(self):
        self.dataset_loader = HumanPoseDSLoader()
        self.dataset_loader.load_data_train_2d_heatmaps()
        return

    def train_heatmaps(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        with tf.Session(config=config) as sess:
            model = pose_hrnet_model.Model()

            # predict 2d pose
            p2d_out_heatmap = model(self.dataset_loader.images_loader, training=True)

            # compute loss
            loss = tf.losses.mean_squared_error(self.dataset_loader.p2d_gt_loader, p2d_out_heatmap)

            learning_rate = tf.placeholder(tf.float32, shape=[])

            # define trainer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch norm
            with tf.control_dependencies(update_ops):
                train_op = tf.train.MomentumOptimizer(learning_rate=TrainingConfig.LEARNING_RATE,
                                                      momentum=0.9).minimize(loss)

            # visualization related
            tf.summary.scalar("mse_loss", loss)
            tf.summary.image("input", tf.expand_dims(self.dataset_loader.images_loader[0], axis=0),
                             max_outputs=4)
            tf.summary.image("heatmap_output",
                             tf.expand_dims(tf.expand_dims(tf.reduce_sum(p2d_out_heatmap[0], axis=2), axis=-1), axis=0))
            tf.summary.image("heatmap_output_neck",
                             tf.expand_dims(tf.expand_dims(p2d_out_heatmap[0][:, :, 10], axis=-1), axis=0))
            tf.summary.image("heatmap_output_hand_right",
                             tf.expand_dims(tf.expand_dims(p2d_out_heatmap[0][:, :, 16], axis=-1), axis=0))
            tf.summary.image("heatmap_output_hand_left",
                             tf.expand_dims(tf.expand_dims(p2d_out_heatmap[0][:, :, 13], axis=-1), axis=0))
            tf.summary.image("heatmap_groundtruth_hand_left",
                             tf.expand_dims(tf.expand_dims(self.dataset_loader.p2d_gt_loader[0][:, :, 16], axis=-1),
                                            axis=0))
            tf.summary.image("heatmap_output_foot_right",
                             tf.expand_dims(tf.expand_dims(p2d_out_heatmap[0][:, :, 3], axis=-1), axis=0))
            tf.summary.image("heatmap_output_foot_left",
                             tf.expand_dims(tf.expand_dims(p2d_out_heatmap[0][:, :, 6], axis=-1), axis=0))
            tf.summary.image("heatmap_groudtruth_foot_left",
                             tf.expand_dims(tf.expand_dims(self.dataset_loader.p2d_gt_loader[0][:, :, 6], axis=-1),
                                            axis=0))
            tf.summary.image("heatmap_groudtruth_foot_right",
                             tf.expand_dims(tf.expand_dims(self.dataset_loader.p2d_gt_loader[0][:, :, 3], axis=-1),
                                            axis=0))
            tf.summary.image("heatmap_groundtruth",
                             tf.expand_dims(tf.expand_dims(tf.reduce_sum(self.dataset_loader.p2d_gt_loader[0], axis=2),
                                                           axis=-1), axis=0))
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(TrainingConfig.LOG_PATH, sess.graph)

            saver = tf.train.Saver(tf.global_variables())
            with tf.name_scope("checkpoint_check"):
                ckpt = tf.train.get_checkpoint_state(TrainingConfig.LOG_PATH)                                            
                try:
                    if ckpt and ckpt.model_checkpoint_path and TrainingConfig.RESUME_TRAINING:
           
                        print("Resume training from {0}".format(tf.train.latest_checkpoint(TrainingConfig.LOG_PATH)))
                        
                        saver.restore(sess, save_path=tf.train.latest_checkpoint(TrainingConfig.LOG_PATH))
                        
                    else:
                    
                        print("Starting a new training session")
                        
                        tf.global_variables_initializer().run()
                except:
                    print("Unable to resume training. Re-initializing weights...")
                    tf.global_variables_initializer().run()
            # training loop
            with tf.name_scope("param_count"):
                trainable_count = tf.reduce_sum(
                    [tf.reduce_prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print("Trainable params: {0}".format(sess.run(trainable_count)))
            with trange(int(TrainingConfig.NUM_EPOCHS * TrainingConfig.NUM_SAMPLES / TrainingConfig.BATCH_SIZE)) as t:
                for i in t:
                    # display training status
                    epoch_cur = i * TrainingConfig.BATCH_SIZE / TrainingConfig.NUM_SAMPLES
                    iter_cur = (i * TrainingConfig.BATCH_SIZE) % TrainingConfig.NUM_SAMPLES
                    t.set_postfix(epoch=epoch_cur,
                                  iter_percent="%d %%" % (iter_cur / float(TrainingConfig.NUM_SAMPLES) * 100))
                    # vis
                    if i % TrainingConfig.LOG_ITER_FREQ == 0:
                        _, summary = sess.run([train_op, merged],
                                              feed_dict={learning_rate: TrainingConfig.LEARNING_RATE})
                        train_writer.add_summary(summary, i)
                    else:
                        _, = sess.run([train_op], feed_dict={learning_rate: TrainingConfig.LEARNING_RATE})

                    # save model
                    if i % TrainingConfig.SAVE_ITER_FREQ == 0:
                        saver.save(sess, os.path.join(TrainingConfig.LOG_PATH, "model"), global_step=i)

            saver.save(sess, os.path.join(TrainingConfig.LOG_PATH, "model"),
                       global_step=int(TrainingConfig.NUM_EPOCHS * TrainingConfig.NUM_SAMPLES /
                                       TrainingConfig.BATCH_SIZE))
