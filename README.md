# Gradient-based Uncertainty for Monocular Depth Estimation

This repository contains the official implementation of our  ECCV 2022 paper. 

## Requirements
We provide the `environment.yml` file with the required packages. The file ca be used to create an Anaconda environment. 

## Datasets
We conduct our evaluations on the datasets [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php). 
[NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) is downloaded as provided by [FastDepth](https://github.com/dwofk/fast-depth) into the folder `nyu_data`. [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) is downloaded according to the instructions from [mono-uncertainty](https://github.com/mattpoggi/mono-uncertainty) into the folder `kitti_data`.


## Pre-trained Models 
We conduct experiments on already trained depth estimation models. The pre-trained models are trained on [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) with monocular and stereo supervision in a self-supervised manner and on [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) in a supervised manner.
In case of [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), we rely on the already trained modes from [mono-uncertainty](https://github.com/mattpoggi/mono-uncertainty). Please follow their instructions to download the respective model weights. 
Our models trained on [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) can be downloaded from the following link: [NYU Models](https://cloudstore.uni-ulm.de/s/7pZn39CTFyMwPBA).
The models can be trained with the following command: 
```
python3 monodepth2/train.py --data_path nyu_data --width 288 --height 224 --max_depth 10 --dataset nyu  
```
To train the log-likelihood maximization model use the additional option `--uncert`- To train the MC Dropout model use the additional option `--dropout`.


## Run Code 
TODO

### Evaluation Supervised 

### Evaluation Self-supervised 

## Reference

## Acknowledgement
We used and modified code parts from the open source projects [monodepth2](https://github.com/nianticlabs/monodepth2) and [mono-uncertainty](https://github.com/mattpoggi/mono-uncertainty). We like to thank the authors for making their code publicly available. 