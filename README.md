# Neural Garment Dynamics via Manifold-Aware Transformers

![Python](https://img.shields.io/badge/Python->=3.8-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=2.0.1-Red?logo=pytorch)

This repository provides the implementation for our manifold-aware transformers, a novel neural network architecture for predicting the dynamics of garments. It is based on our work [Neural Garment Dynamics via Manifold-Aware Transformers](https://peizhuoli.github.io/manifold-aware-transformers/index.html) that is published in EUROGRAPHICS 2024.

<img src="https://peizhuoli.github.io/manifold-aware-transformers/images/video_teaser_small.gif" slign="center">


## Prerequisites

This code has been tested under Ubuntu 20.04. Before starting, please configure your Anaconda environment by
~~~bash
conda env create -f environment.yml
conda activate manifold-aware-transformers
~~~


Alternatively, you may install the following packages (and their dependencies) manually:

- numpy == 1.23.1 (note `numpy.bool` is deprecated in higher version and it causes an error when loading SMPL model)
- pytorch == 2.0.1
- scipy >= 1.10.1
- cholespy == 1.0.0
- scikit-sparse == 0.4.4
- libigl == 2.4.1
- tensorboard >= 2.12.1
- tqdm >= 4.65.0
- chumpy == 0.70


## Quick Start

We provide several pre-trained models trained on different datasets. Download the pre-trained models and the example sequences from [Google Drive](https://drive.google.com/drive/folders/1tjvFg6ymHUyjxpKmVs_U8ysBhK6ILEA_?usp=share_link). Please extract the pre-trained models and example sequences, and put them under the `pre-trained` and `data` directory directly under the root of the project directory, respectively.

### Garment Prediction

Run `demo.sh`.

The prediction of the network will be stored in `[path to pre-trained model]/sequence/prediction.pkl`. The corresponding ground truth and body motion are stored in `gt.pkl` and `body.pkl` respectively. Please refer to [here](#mesh-sequence-format-and-visualization) for the specification and visualization of the predicted mesh sequence.

### Mesh Sequence Format and Visualization

We use a custom format to store a sequence of meshes. The specific format can be found in the function `write_vert_pos_pickle()` in `mesh_utils.py`.

We provide a plugin for visualizing the mesh sequences directly in Blender [here](https://github.com/PeizhuoLi/STOP-motion-OBJ/). It is based on the [STOP-motion-OBJ](https://github.com/neverhood311/Stop-motion-OBJ) plugin by [@neverhood311](https://github.com/neverhood311).

### Evaluation

We provide a [small sample](https://drive.google.com/drive/folders/1Mp-wmlU3B-J47SoJU9et1-Z1xkt0iLeX?usp=share_link) of pre-processed VTO and CLOTH3D datasets for reproducing our quantitative evaluations. Please download and extract the sample data to the `data` directory directly under the root of the project directory.

Use the following command to calculate the mean vertex error of the pre-trained models on VTO dataset:

~~~bash
python evaluate.py --dataset=vto
~~~

and for CLOTH3D dataset:

~~~bash
python evaluate.py --dataset=cloth3d
~~~

Due to the nondeterministic algorithms used in Pytorch, the results may differ in each run, and may also slightly differ from the numbers reported in the paper.

## Data Preprocessing

The input garment geometry is decimated to improve the running efficiency. A separate [module](https://github.com/PeizhuoLi/mesh_simplifier/) implemented with C++ is *required*. 

Our model requires the signed distance function from the garment geometry to the body geometry as input. It is calculated on the fly for inference time using a highly optimized GPU implementation. 

The current implementation for inference requires the same format, and you may use the following steps to preprocess the data for inference as well.

### VTO dataset

Please download the VTO dataset from [here](https://github.com/isantesteban/vto-dataset), and run the following command to preprocess the data:

~~~bash
python parse_data_vto.py --data_path_prefix=[path to downloaded vto dataset] --save_path=[path to save the preprocessed data]
~~~

### Cloth3D dataset

Please download the CLOTH3D dataset from [here](https://chalearnlap.cvc.uab.cat/dataset/38/description/#), put the `DataReader` directory from the official [starter kit](http://158.109.8.102/CLOTH3D/StarterKit.zip) under `data/cloth3d`, and run the following command to preprocess the data:

~~~bash
python parse_data_cloth3d.py --data_path_prefix=[path to downloaded cloth3d dataset] --save_path=[path to save the preprocessed data]
~~~


## Train from scratch

Coming soon.

## Acknowledgments

The code in `dataset/smpl.py` is adapted from [SMPL](https://github.com/CalciferZh/SMPL) by [@CalciferZh](https://github.com/CalciferZh).


## Citation

If you use this code for your research, please cite our paper:

~~~bibtex
@article{Li2024NeuralGarmentDynamics,
  author  = {Li, Peizhuo and Wang, Tuanfeng Y. and Kesdogan, Timur Levent and Ceylan, Duygu and Sorkine-Hornung, Olga},
  title   = {Neural Garment Dynamics via Manifold-Aware Transformers},
  journal = {Computer Graphics Forum (Proceedings of EUROGRAPHICS 2024)},
  volume  = {43},
  number  = {2},
  year    = {2024},
}
~~~
