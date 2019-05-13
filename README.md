## Position-Focused-Attention-Network
Position Focused Attention Network for Image-Text Matching

## Introduction

This is Position Focused Attention Network, source code of position attention for Image-Text Matching (project page) from Tencent. It is built on top of the SCAN (by kuanghuei) in PyTorch.

## Requirements and Installation
We recommended the following dependencies.

* Python 2.7
* [PyTorch](http://pytorch.org/) 0.3
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)


## Download data
Download the dataset files. We use the dataset files created by SCAN [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN). The position information of images can be downloaded from XX (for Flickr30K) and XX (for MS-COCO).

```bash
wget https://XX
wget https://XX
```

## Training new models
Run run_train.sh
