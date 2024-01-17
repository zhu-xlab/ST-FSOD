# ST-FSOD
Official implementation of TGRS paper: Few-shot Object Detection in Remote Sensing: Lifting the Curse of Incompletely Annotated Novel Objects

## Prepare the Datasets
### Install Dataset4EO
We use Dataset4EO to handle our data loading. Dataset4EO is a composable data loading package based on TorchData. More information can be found at https://github.com/EarthNets/Dataset4EO.

```shell
git clone https://github.com/EarthNets/Dataset4EO.git
cd Dataset4EO
sh install_requirements.sh
pip install -e .
```

### DIOR datasets
1. Download the dataset at https://gcheng-nwpu.github.io/#Datasets
2. Put the four download .zip files in /path/to/your_datasets/DIOR
3. Set the path to the datasets in the configuration files under the folder ./configures/\_base_/datasets/fine_tune_based/. The files to be set include base_dior_bs16.py, few_shot_dior_bs8.py, base_dior-trainval_bs16.py, and few_shot_dior-trainval_bs8.py
```shell
datapipe_root = 'path/to/your_datasets/DIOR'
```

### iSAID datasets
1. Download the datset and preprocess the data (basically seperate the images into patches) following the instructions of the official development kit at https://github.com/CAPTAIN-WHU/iSAID_Devkit
2. Make sure you end up with a data folder at /path/to/your_datasets/iSAID_patches
3. Set the paths to the datasets in the configuration files like we did for DIOR dataset. The files to be set include base_isaid_bs16.py and few_shot_isaid_1k.py.

### NWPU-VHR10.v2 datasets
TODO: add support to NWPU-VHR10.v2 in Dataset4EO

## Perform Base Training
1. Train the base model using script:
```shell
python tools/detection/train.py configs/st_fsod/{dataset_name}/split1/st-fsod_maskrcnn_r101_40k_{dataset_name}-split{split_number}_base-training.py
```
3. Change the dataset_name and split_number accordingly. dataset_name should be one of "dior", "isaid" and "nwpu"

## Perform Few-shot Fine-tuning
1. Convert the trained base model by initializing the bounding box head:
```shell
python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/st-fsod_maskrcnn_r101_40k_{dataset_name}-split{split_number}_base-training/iter_{iter_number}.pth --method random_init --save-dir work_dirs/temp/ --{dataset_name}
```
replace the dataset_name, split_number and iter_number of the checkpoint accordingly.

## Evaluation


## Acknowledgement
https://github.com/EarthNets
https://github.com/open-mmlab/mmfewshot
https://github.com/open-mmlab/mmdetection
