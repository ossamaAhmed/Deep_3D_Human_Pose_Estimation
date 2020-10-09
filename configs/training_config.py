import os
from configs.master_config import CONFIG


class TrainingConfig(object):
    NUM_SAMPLES = 312188
    # Hyper parameters
    NUM_EPOCHS = CONFIG.TRAIN_EPOCHS
    BATCH_SIZE = CONFIG.TRAIN_BATCHSIZE
    LEARNING_RATE = 0.003
    LOG_ITER_FREQ = 1
    SAVE_ITER_FREQ = 2000
    LOG_PATH = CONFIG.TRAIN_LOG_PATH
    RESUME_TRAINING = True
    CHECKPOINT_DIR = os.path.join(LOG_PATH, "model")
