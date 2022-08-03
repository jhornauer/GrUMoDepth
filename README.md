# Gradient-based Uncertainty for Monocular Depth Estimation

This repository contains the official implementation of our  ECCV 2022 paper. 

![Overview](images/github_overview.jpg)

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
To train the log-likelihood maximization model use the additional option `--uncert`. To train the MC Dropout model use the additional option `--dropout`.


## Run Code 
We conduct our experiments on [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) (self-supervised) and [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) (supervised) data. 

As explained in our paper, we apply our training-free uncertainty estimation method to already trained models. Therefore, we have different base models and compare the uncertainty estimation approaches on the base models.

### Evaluation Self-Supervised
In the self-supervised case, we have the base models MC Dropout (Drop), Bootstrapped Ensembles (Boot), Post-Processing (Post), Log-likelihood Maximization (Log) and Self-Teaching (Self).
We compare the post hoc uncertainty estimation approaches on *Post*, *Log* and *Self*. As post hoc uncertainty estimation approaches we consider the variance over different test-time augmentations (Var), inference-only dropout (In-Drop) and our approach.

For the evaluation of the Drop model and the Boot model as well as the Log and the Self base models you can refer to [mono-uncertainty](https://github.com/mattpoggi/mono-uncertainty).

#### Evaluation of our gradient-based uncertainty estimation method:
For the evaluation of our method on the Post base model run: 
```
python3 generate_maps.py --data_path kitti_data --load_weights_folder weights/S/Monodepth2-Post/models/weights_19/ --eval_split eigen_benchmark --eval_stereo --output_dir experiments/S/post_model/Grad --grad
python3 evaluate.py --ext_disp_to_eval experiments/S/post_model/Grad/raw/ --eval_stereo --max_depth 80 --eval_split eigen_benchmark --eval_uncert --output_dir experiments/S/post_model/Grad --grad
```

For the evaluation of our method on the Log base model run: 
```
python3 generate_maps.py --data_path kitti_data --load_weights_folder weights/S/Monodepth2-Log/models/weights_19/ --eval_split eigen_benchmark --eval_stereo --output_dir experiments/S/log_model/Grad --uncert --grad --w 2.0 
python3 evaluate.py --ext_disp_to_eval experiments/S/log_model/Grad/raw/ --eval_stereo --max_depth 80 --eval_split eigen_benchmark --eval_uncert --output_dir experiments/S/log_model/Grad --grad --uncert --w 2.0
```

For the evaluation of our method on the Self base model run: 
```
python3 generate_maps.py --data_path kitti_data --load_weights_folder weights/S/Monodepth2-Self/models/weights_19/ --eval_split eigen_benchmark --eval_stereo --output_dir experiments/S/self_model/Grad --uncert --grad --w 2.0 
python3 evaluate.py --ext_disp_to_eval experiments/S/self_model/Grad/raw/ --eval_stereo --max_depth 80 --eval_split eigen_benchmark --eval_uncert --output_dir experiments/S/self_model/Grad --grad --uncert --w 2.0
```

To change the decoder layer for the gradient extraction use the argument `--ext_layer` with values between `0` and `10`.

To change the augmentation for the generation of the reference depth use the argument `--gref` with one of the values: `flip`, `gray` , `noise` or `rot`. 

##### Evaluation of In-Drop method: 
For the evaluation of the In-Drop method on the Post base model run: 
```
python3 generate_maps.py --data_path kitti_data --load_weights_folder weights/S/Monodepth2-Post/models/weights_19/ --eval_split eigen_benchmark --eval_stereo --output_dir experiments/S/post_model/In-Drop --infer_dropout
python3 evaluate.py --ext_disp_to_eval experiments/S/post_model/Infer-Drop/raw/ --eval_stereo --max_depth 80 --eval_split eigen_benchmark --eval_uncert --output_dir experiments/S/post_model/In-Drop --infer_dropout
```

To change the dropout probability use the argument `--infer_p` with values between `0.0` and `1.0`. Default ist `0.2`. 


##### Evaluation of the Var method: 
For the evaluation of the Var method on the Post base model run: 
```
python3 generate_maps.py --data_path kitti_data --load_weights_folder weights/S/Monodepth2-Post/models/weights_19/ --eval_split eigen_benchmark --eval_stereo --output_dir experiments/S/post_model/Var --var_aug
python3 evaluate.py --ext_disp_to_eval experiments/S/post_model/Var/raw/ --eval_stereo --max_depth 80 --eval_split eigen_benchmark --eval_uncert --output_dir experiments/S/post_model/Var --var_aug
```

For the evaluation of the models trained with monocular supervision replace the folder `S` with `M` and the argument `--eval_stereo` with `--eval_mono`. 

### Evaluation Supervised
In the supervised case, we have the base models MC Dropout (Drop), Post-Processing (Post) and Log-likelihood Maximization (Log).
We compare the post hoc uncertainty estimation approaches on *Post* and *Log*. As post hoc uncertainty estimation approaches we consider the variance over different test-time augmentations (Var), inference-only dropout (In-Drop) and our approach.

#### Evaluation of our gradient-based uncertainty estimation method: 
For the evaluation of our method on the Post base model run: 
```
python3 evaluate_supervised.py --max_depth 10 --load_weights_folder weights/NYU/Monodepth2/weights/ --data_path nyu_data --eval_uncert --output_dir experiments/NYU/post_model/Grad/ --grad 
```

For the evaluation of our method on the Log base model run: 
```
python3 evaluate_supervised.py --max_depth 10 --load_weights_folder weights/NYU/Monodepth2-Log/weights/ --data_path nyu_data --eval_uncert --output_dir experiments/NYU/log_model/Grad/ --grad --uncert -w 2.0 
```

##### Evaluation of the Drop model: 
For the evaluation of the Drop model run: 
```
python3 evaluate_supervised.py --max_depth 10 --load_weights_folder weights/NYU/Monodepth2-Drop/weights/ --data_path nyu_data --eval_uncert --output_dir experiments/NYU/drop_model/Drop/ --dropout
```

##### Evaluation of In-Drop method: 
For the evaluation of the In-Drop method on the Post base model run: 
```
python3 evaluate_supervised.py --max_depth 10 --load_weights_folder weights/NYU/Monodepth2/weights/ --data_path nyu_data --eval_uncert --output_dir experiments/NYU/post_model/In-Drop/ --infer_dropout
```

To change the dropout probability use the argument `--infer_p` with values between `0.0` and `1.0`. Default ist `0.2`. 


##### Evaluation of the Var method: 
For the evaluation of the Var method on the Post base model run: 
```
python3 evaluate_supervised.py --max_depth 10 --load_weights_folder weights/NYU/Monodepth2/weights/ --data_path nyu_data --eval_uncert --output_dir experiments/NYU/post_model/Var/ --var_aug
```

## Reference
TODO

## Acknowledgement
We used and modified code parts from the open source projects [monodepth2](https://github.com/nianticlabs/monodepth2) and [mono-uncertainty](https://github.com/mattpoggi/mono-uncertainty). We like to thank the authors for making their code publicly available. 