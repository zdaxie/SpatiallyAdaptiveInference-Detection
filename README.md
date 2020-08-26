# Spatially Adaptive Inference with Stochastic Feature Sampling and Interpolation

## Introduction

This repo is the official implementation of ["Spatially Adaptive Inference with Stochastic Feature Sampling and Interpolation"](https://arxiv.org/abs/2003.08866) on COCO object detection. The code is based on [MMDetection](https://github.com/open-mmlab/mmdetection) v0.6.0.

<div align="center">
    <img src="demo/github_raw_image.png" width="200px" /><img src="demo/github_deterministic_sampling.png" width="200px" /><img src="demo/github_stochastic_sampling.png" width="200px" />
    <p>Deterministic sampling(middle) and stochastic sampling(right) from the raw image(left).</p>
</div>

<div align="center">
    <img src="demo/github_pipeline_gumbel.png" width="500px" />
    <p>Stochastic sampling-interpolation network(a) and comparison between deterministic sampling(b, left) and stochastic sampling(b, right).</p>
</div>

**Abstract.** In the feature maps of CNNs, there commonly exists considerable spatial redundancy that leads to much repetitive processing. Towards reducing this superﬂuous computation, we propose to compute features only at sparsely sampled locations, which are probabilistically chosen according to activation responses, and then densely reconstruct the feature map with an eﬃcient interpolation procedure. With this sampling-interpolation scheme, our network avoids expending computation on spatial locations that can be eﬀectively interpolated, while being robust to activation prediction errors through broadly distributed sampling. A technical challenge of this sampling-based approach is that the binary decision variables for representing discrete sampling locations are non-diﬀerentiable, making them incompatible with backpropagation. To circumvent this issue, we make use of a reparameterization trick based on the Gumbel-Softmax distribution, with which backpropagation can iterate these variables towards binary values. The presented network is experimentally shown to save substantial computation while maintaining accuracy over a variety of computer vision tasks.

## Model Zoo

All the models are based on the Faster R-CNN with FPN.

|Backbone|Resolution|Sparse Loss Weight|mAP|GFlops|Config|Model URL|Model sha256sum|
|-----|-----|-----|-----|-----|-----|-----|-----|
|ResNet-101| 500|-|38.5| 70.0|[config](./configs/resnet_101_faster_rcnn_res500.py) |[model](https://drive.google.com/file/d/1QP1s5EJ3ld5H0tCqW_XatSOHdf7vpO7E/view?usp=sharing)|206b4c0e|
|ResNet-101| 600|-|40.4|100.2|[config](./configs/resnet_101_faster_rcnn_res600.py) |[model](https://drive.google.com/file/d/1gPITSogNwTwbPdA6Cbez4rmjrq2ENSSi/view?usp=sharing)|c4e102de|
|ResNet-101| 800|-|42.3|184.1|[config](./configs/resnet_101_faster_rcnn_res800.py) |[model](https://drive.google.com/file/d/1U_-7b9a2VMM81IJu4-gEyGtsL7lupfAa/view?usp=sharing)|3fc2af7a|
|ResNet-101|1000|-|43.4|289.5|[config](./configs/resnet_101_faster_rcnn_res1000.py)|[model](https://drive.google.com/file/d/1XOyzBgNndl8TKaMs0_5QSu-obHscWsWp/view?usp=sharing)|e043c999|
|SparseResNet-101|1000|0.02|43.3|164.8|[config](./configs/sparse_resnet_101_faster_rcnn_sparse_loss_weight_0_02.py)|[model](https://drive.google.com/file/d/1z942uPLEsfa0Cra7C63qbeWXU7FMURyS/view?usp=sharing)|16a152e0|
|SparseResNet-101|1000|0.05|42.7|120.3|[config](./configs/sparse_resnet_101_faster_rcnn_sparse_loss_weight_0_05.py)|[model](https://drive.google.com/file/d/16X-IaqlJwHhZnbVMOdDeXkNfICNCL3Dw/view?usp=sharing)|f0a467c8|
|SparseResNet-101|1000| 0.1|41.9| 94.4|[config](./configs/sparse_resnet_101_faster_rcnn_sparse_loss_weight_0_1.py) |[model](https://drive.google.com/file/d/1KrCMXTvSJR7QslsfXRnZCQJUvtCyS8SW/view?usp=sharing)|1c9bf665|
|SparseResNet-101|1000| 0.2|40.7| 71.4|[config](./configs/sparse_resnet_101_faster_rcnn_sparse_loss_weight_0_2.py) |[model](https://drive.google.com/file/d/1El9uroSGNjRAixNpOAAjczTyKGa9HISu/view?usp=sharing)|46044e4a|

## Getting Started

### Requirements

At present, we have not checked the compatibility of the code with other versions of the packages, so we only recommend the following configuration.

- Python 3.7
- PyTorch == 1.1.0
- Torchvision == 0.3.0
- CUDA 9.0
- Other dependencies

### Installation

We recommand using conda env to setup the experimental environments.

```shell script
# Create environment
conda create -n SAI_Det python=3.7 -y
conda activate SAI_Det

# Install PyTorch & Torchvision
conda install pytorch=1.1.0 cudatoolkit=9.0 torchvision -c pytorch -y

# Clone repo
git clone https://github.com/zdaxie/SpatiallyAdaptiveInference-Detection ./SAI_Det
cd ./SAI_Det

# Create soft link for data
mkdir data
cd data
ln -s ${COCO-Path} ./coco
cd ..

# Install requirements and Compile operators
./init.sh
```

### Running

For now, we only support training with 8 GPUs.

```shell script
# Test with the given config & model
./tools/dist_test.sh ${config-path} ${model-path} ${num-gpus} --out ${output-file.pkl}

# Train with the given config
./tools/dist_train.sh ${config-path} ${num-gpus}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you use our codebase or models in your research, please cite this project.

```
@InProceedings{xie2020spatially,
author = {Xie, Zhenda and Zhang, Zheng and Zhu, Xizhou and Huang, Gao and Lin, Steve},
title = {Spatially Adaptive Inference with Stochastic Feature Sampling and Interpolation},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2020},
month = {August},
}
```
