from yacs.config import CfgNode as CN

CONFIG = CN()
CONFIG.DATAPATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"

CONFIG.TRAIN_MODEL = 'highresnet'
CONFIG.TRAIN_BATCHSIZE = 2
CONFIG.TRAIN_EPOCHS = 30
CONFIG.TRAIN_LOG_PATH = "./log/highresnet/"
CONFIG.TRAIN_FLAG = False

CONFIG.TEST_BATCHSIZE = 32
CONFIG.TEST_HIGHRESNET_FLAG = True
CONFIG.TEST_BASELINE_REGRESSOR_FLAG = True
CONFIG.TEST_HIGHRESNET_MODEL_PATH = "/cluster/scratch/oahmed/log/highresnet_no_flip/"
CONFIG.TEST_BASELINE_REGRESSOR_MODEL_PATH = "/cluster/scratch/oahmed/log/regressor_g_no_flip/"
CONFIG.TEST_HIGHRESNET_CHECKPOINT = "model-94000"
CONFIG.TEST_BASELINE_REGRESSOR_CHECKPOINT = "model-60974"