# ST-FSOD
Official implementation of TGRS paper: Few-shot Object Detection in Remote Sensing: Lifting the Curse of Incompletely Annotated Novel Objects

## Prepare the Datasets
#### Install Dataset4EO
We use Dataset4EO to handle our data loading. Dataset4EO is a composable data loading package based on TorchData. More information can be found at https://github.com/EarthNets/Dataset4EO.

```shell
git clone https://github.com/EarthNets/Dataset4EO.git
cd Dataset4EO
sh install_requirements.sh
pip install -e .
```

#### DIOR datasets
1. Download the dataset at https://gcheng-nwpu.github.io/#Datasets
2. Put the four download .zip files in /path/to/your_datasets/DIOR
3. Set the path to the datasets in the configuration files under the folder ./configures/\_base_/datasets/fine_tune_based/. The files to be set include base_dior_bs16.py, few_shot_dior_bs8.py, base_dior-trainval_bs16.py, and few_shot_dior-trainval_bs8.py
```shell
datapipe_root = 'path/to/your_datasets/DIOR'
```

#### iSAID datasets
1. Download the datset and preprocess the data (basically seperate the images into patches) following the instructions of the official development kit at https://github.com/CAPTAIN-WHU/iSAID_Devkit
2. Make sure you end up with a data folder at /path/to/your_datasets/iSAID_patches
3. Set the paths to the datasets in the configuration files like we did for DIOR dataset. The files to be set include base_isaid_bs16.py and few_shot_isaid_1k.py.

#### NWPU-VHR10.v2 datasets
TODO: add support to NWPU-VHR10.v2 in Dataset4EO

## Train the models
#### Perform Base Training
1. Train the base model using script:
```shell
python tools/detection/train.py configs/st_fsod/dior/split1/st-fsod_maskrcnn_r101_40k_dior-split1_base-training.py
```

2. Change the dataset_name and split_number accordingly. dataset_name should be one of "dior", "isaid" and "nwpu". Note that for DIOR dataset, the split1 in the configuration file corresponds to the split proposed in paper "Few-shot object detection on remote sensing images". Split2-5 correspond to the four splits proposed in "Prototype-cnn for few-shot object detection in remote sensing images".

#### Perform Few-shot Fine-tuning
1. Convert the trained base model by initializing the bounding box head (this is required due to the fact that we are following the two-stage fine-tuning process proposed in the TFA paper):
```shell
python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/st-fsod_maskrcnn_r101_40k_dior-split1_base-training/iter_40000.pth --method random_init --save-dir work_dirs/path/to/your/model.pth --dior
```
Replace the dataset name, split number and iteration number of the checkpoint of the base model accordingly (the checkpoint to be used can be selected according to the validation results in the training log).

2. set the path to the converted base model in the configuration file at configs/st_fsod/dior/split1/seed0/st-fsod/st-fsod_maskrcnn_r101_dior-split1_seed0_3shot-fine-tuning.py
```shell
load_from = ('work_dirs/path/to/your/model.pth')
```
One may need to change the split number, seed number, dataset name and number shots accordingly.

3. Train the fine-tuned models:
```shell
python tools/detection/train.py configs/st_fsod/dior/split1/seed0/st-fsod/st-fsod_maskrcnn_r101_dior-split1_seed0_3shot-fine-tuning.py
```

4. If you would like to use a different seed number, e.g., 42, simply modifying the corresponding configuration file:
```shell
seed = 42
```

## Evaluation
#### Evaluate the base model
```shell
python tools/detection/test.py configs/st_fsod/dior/split1/st-fsod_maskrcnn_r101_40k_dior-split1_base-training.py --work-dir work_dirs/st-fsod_maskrcnn_r101_40k_dior-split1_base-training/iter_40000.pth
```
Change the dataset name, split number and iteration number of the checkpoint accordingly.

#### Evaluate the fine-tuned model
The evaluation script has the same format as above:
```shell
python tools/detection/test.py configs/st_fsod/dior/split1/seed0/st-fsod/st-fsod_maskrcnn_r101_dior-split1_seed0_3shot-fine-tuning.py --work-dir work_dirs/st-fsod_maskrcnn_r101_dior-split1_seed0_3shot-fine-tuning/iter_1000.pth --eval='mAP'
```
Note that one should choose the checkpoints to be evaluated according to the validation results (which can be seen in the training log). We noticed that the variances of the accuracies among different checkpoints are quite large, which indicates the validation process is neccessary.

## Acknowledgement
https://github.com/EarthNets

https://github.com/open-mmlab/mmfewshot

https://github.com/open-mmlab/mmdetection
