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
3. Set the path to the datasets in the configuration files in ./configures/\_base_/datasets/fine_tune_based/. The files to be set include base_dior_bs16.py, few_shot_dior_bs8.py, base_dior-trainval_bs16.py, and few_shot_dior-trainval_bs8.py
```shell
datapipe_root = 'path/to/your_datasets/DIOR'
```
