from train.train_model import TrainModel
from train.train_regressor import TrainRegressor
from validate.validate_model import Validate
from configs.master_config import CONFIG
import tensorflow as tf


def main():
    if CONFIG.TRAIN_FLAG:
        if CONFIG.TRAIN_MODEL == 'highresnet':
            train_model = TrainModel()
            train_model.train_heatmaps()
        elif CONFIG.TRAIN_MODEL == 'regressor':
            train_model = TrainRegressor()
            train_model.train()
        else:
            raise NameError("Invalid model name. Got {0} but expected one of: 'highresnet', 'regressor'".format(
                CONFIG.TRAIN_MODEL))

    else:
        if CONFIG.TEST_HIGHRESNET_FLAG:
            test_model = Validate(two_d_input=False,
                                  two_d_pose_file=None,
                                  submission_file_name="submission")
            test_model.validate_model_public_highresnet_2d_pose()
    
        if CONFIG.TEST_BASELINE_REGRESSOR_FLAG:
            tf.reset_default_graph()
            test_model = Validate(two_d_input=True,
                                  two_d_pose_file="submission.csv",
                                  submission_file_name="final_submission")
            test_model.validate_model_public_baseline_3d_regressor()


if __name__ == "__main__":
    main()
