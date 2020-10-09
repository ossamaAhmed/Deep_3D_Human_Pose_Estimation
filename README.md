# 3D Human Pose Estimation Challenge

This repository contains to the final submission for the **3D Human Pose Estimation Challenge** (Task 2) for the course on Machine Perception 2019 at ETH Zurich. 
We tackle this problem by means of a two-component deep learning pipeline, consisting on a re-implementation of the HRNet by Ke Sun et.al. (2019) combined with the dense model proposed by Martinez et.al (2017) for lifting 2D coordinates to 3D. 

This repository is prepared to work with the Human3.6M dataset, which contains images of a single subject with annotations of 2D and 3D positions of 17 human body joints.

## Installation

### Prerequisites

* Python 3.5, 3.6 or 3.7
* Tensorflow GPU 1.12 or 1.13 properly installed

### Repository setup

Go to the desired folder and clone this repository. Install the required python packages:
```
git clone https://gitlab.inf.ethz.ch/-/ide/project/COURSE-MP19/SuperkondiHeroes/
cd SuperkonidHeroes
python setup.py install --user
```

### Dataset

Finally, make sure that the dataset is structured according to the default configuration (there should be a total of 312188 images in the image directory).

    └── /dataset_dir/
        ├── annot/
        |   ├── mean.txt
        |   ├── std.txt
        |   ├── train.h5
        |   ├── train_images.txt
        |   └── valid_images.txt
        └── images/
            ├── ...
            ├── S6_Directions.54138969_002621.jpg
            ├── S6_Directions.54138969_002626.jpg
            └── ...
             
And set-up the dataset configuration variables in [__master_config.py__](./configs/master_config.py):

``` 
CONFIG.DATAPATH = /path/to/datset_dir/
```

## Training the models

The two models must be trained separately. These are the instructions to reproduce our training sessions:

### Train HRNet

Set up the training configuration flags in [__master_config.py__](./configs/master_config.py):

```
CONFIG.TRAIN_MODEL = 'highresnet'
CONFIG.TRAIN_BATCHSIZE = 32
CONFIG.TRAIN_EPOCHS = 30
CONFIG.TRAIN_LOG_PATH = "/path/to/log/dir/hrnet/"
CONFIG.TRAIN_FLAG = True
```

Run the main script in a CUDA environment:

```
python main.py
```

Expected training time on 1 Nvidia GTX 1080 GPU: 

### Train 2D to 3D regressor

Set up the training configuration flags in [__master_config.py__](./configs/master_config.py):

```
CONFIG.TRAIN_MODEL = 'regressor'
CONFIG.TRAIN_BATCHSIZE = 512
CONFIG.TRAIN_EPOCHS = 180
CONFIG.TRAIN_LOG_PATH = "/path/to/log/dir/regressor/"
CONFIG.TRAIN_FLAG = True
```

Run the main script. This training is fast enough to be efficiently performed on a CPU:

```
python main.py
```

Expected training time on 4 Intel Xeon E5-2697v4 CPU's (total RAM = 32GB): 24 - 28h

The respective checkpoints for the two models will be saved in the specified Log directories under the names `model-#####.index` and `model-#####.data-00000-of-00001`, where `#####` is an automatically generated increasing number corresponding to the training step. 

## Testing the models

Set up the testing configuration flags in [__master_config.py__](./configs/master_config.py).
For testing the entire pipeline, set both `TEST_HIGHRESNET_FLAG` and `TEST_BASELINE_REGRESSOR_FLAG` to `True`. Otherwise set to `False` one of them.

```
CONFIG.TRAIN_FLAG = False

CONFIG.TEST_BATCHSIZE = 32
CONFIG.TEST_HIGHRESNET_FLAG = True
CONFIG.TEST_BASELINE_REGRESSOR_FLAG = True
CONFIG.TEST_HIGHRESNET_MODEL_PATH = "/path/to/log/dir/hrnet/"
CONFIG.TEST_BASELINE_REGRESSOR_MODEL_PATH = "/path/to/log/dir/regressor/"
CONFIG.TEST_HIGHRESNET_CHECKPOINT = "name_of_last_hrnet_checkpoint"              # e.g. "model-94000"
CONFIG.TEST_BASELINE_REGRESSOR_CHECKPOINT = "name_of_last_regressor_checkpoint"  # e.g. "model-60974"
```

Run the main script.

```
python main.py
```

## Authors

* **Ossama Ahmed** - oahmed@student.ehtz.ch
* **Guillem Torrente** - tguillem@student.ethz.ch
