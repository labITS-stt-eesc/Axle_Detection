# Axle_Detection

![Keywords](https://img.shields.io/badge/Keywords-Neural%20Networks%2C%20Python%2C%20Transport%20Engineering-blueviolet?style=flat-square) ![GitHub](https://img.shields.io/github/license/leandromarcomini/Axle_Detection?style=flat-square) ![GitHub](https://img.shields.io/github/languages/top/leandromarcomini/Axle_Detection?style=flat-square)


Compilation of neural networks used for training and detecting vehicle axles. The dataset used in all networks is available to [donwload](https://doi.org/10.5281/zenodo.5744737) in Zenodo, and it's composed of 725 images of trucks from a sidepoint view, to enable axle detection. If the dataset or this repo happens to be usefull for anyone, please cite us where relevant, using any of these publications:

<a id="1">[1]</a> 
Marcomini, L. A., Cunha, A. L. (2022). 
Truck Axle Detection with Convolutional Neural Networks.
arXiv preprint [arXiv:2204.01868](https://arxiv.org/abs/2204.01868)

<a id="2">[2]</a> 
Marcomini, L. A., Cunha, A. L. (2021).
Truck Image Dataset (1.0.0)
Zenodo. [Data set](https://doi.org/10.5281/zenodo.5744737)


## Conda Environment
YAML files are provided to facilitate the creation of environments using conda (or any python env manager that supports YAML files), and are located in the root directory of this repo. To clone the environment, simply run

```
conda env create --name <env_name> --file <env_file.yml>
```

Conda will try to download and parse all libraries used on each environment. To check which packages are used, you can open the .yml files using your preferred text editor (which should be [Notepad++](https://notepad-plus-plus.org/downloads/), just saying).

## YOLOv3

Required versions:
- Keras: 2.4.3
- Tensorflow-gpu: 2.2.0


## SSD

Required versions:
- Keras: 2.4.3
- Tensorflow-gpu: 2.2.0


## Faster R-CNN

Required versions:
- Keras: 2.3.1
- Tensorflow-gpu: 2.7.0


## GPU libs (GTX1080 & GTX1060)
- CUDA 10.1
- cuDNN 7.6.5.32 for CUDA 10.1
- Visual Studio Community 2019 - Workload: Desktop development with C++
