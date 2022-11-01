# README

This folder contains the implementation of **Amplifying Membership Exposure via Data Poisoning**. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset
1. MNIST (part of TensorFlow)
2. CIFAR-10 (part of TensorFlow)
3. [STL-10](https://cs.stanford.kedu/~acoates/stl10/). Please download the dataset from [the official link](http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz) and uncompress the downloaded file (will get the `stl10_binary` folder). Then move the folder `stl10_binary` into the `dataset`.
4. [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Please download `img_align_celeba` and `img_align_celeba.csv` from [this link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset). Then, create a subfolder `celeba` under `dataset`, and move the file `list_attr_celeba.csv` and the folder `img_align_celeba` into `celeba`.
5. [PatchCamelyon](https://patchcamelyon.grand-challenge.org/). Please download `camelyonpatch_level_2_split_test_x.h5` and `camelyonpatch_level_2_split_test_y.h5` from [this link](https://github.com/basveeling/pcam). Then, create a subfolder `pathcamelyon` under `dataset`, and move the two files into `patchcamelyon`.

The structure of the `dataset` folder should be
```
dataset
+-- stl10_binary
|   +-- train_X.bin
|   +-- train_Y.bin
|   +-- test_X.bin
|   +-- test_Y.bin
|   ...
+-- celeba
|   +-- img_align_celeba
|   +-- list_attr_celeba.csv
+-- patchcamelyon
|   +-- camelyonpatch_level_2_split_test_x.h5
|   +-- camelyonpatch_level_2_split_test_y.h5
```

## Basic Usage
File `poisoning_attack.py` allows to run our poisoning attacks.

Example:
```
python poisoning_attack.py --target_class 0 \
                           --dataset cifar10 \
                           --encoder xception \
                           --seed_amount 1000 \
                           --attack_type clean_label \
                           --device_no 0
```

## Example Attack 

We provide example attacks in `attack_example.sh`. Directly run:
```
./attack_example.sh
```

## Evaluation
To evaluate our poisoning attacks, first run:
```
./poisoning_models.sh
```
to generate poisoning datasets and poisoned models.

Then, run:
```
./evaluate_attack.sh
```
to get the evaluation results.

## Citation
```
@inproceedings{CSSWZ22,
author = {Yufei Chen and Chao Shen and Yun Shen and Cong Wang and Yang Zhang},
title = {{Amplifying Membership Exposure via Data Poisoning}},
booktitle = {{Annual Conference on Neural Information Processing Systems (NeurIPS)}},
publisher = {NeurIPS},
year = {2022}
}
```
