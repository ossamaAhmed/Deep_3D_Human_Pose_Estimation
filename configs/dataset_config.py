from configs.master_config import CONFIG
import os


class DatasetConfig(object):
    DATA_PATH = CONFIG.DATAPATH
    TRAINBATCH_SIZE = CONFIG.TRAIN_BATCHSIZE
    TESTBATCH_SIZE = CONFIG.TEST_BATCHSIZE
    STATISTICS_PATH = "./statistics"
    POSE_2D_MEAN_PATH = os.path.join(STATISTICS_PATH, "mean_2d")
    POSE_2D_STD_PATH = os.path.join(STATISTICS_PATH, "std_2d")
