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
The Tencent-News dataset files can be downloaded from XX.

```bash
wget https://XX
wget https://XX
```

## Training new models
Run run_train.sh

Arguments used to train Flickr30K models and MS-COCO models are as same as those of SCAN:

For Flickr30K:
| Method    | Arguments |
| :-------: | :-------: |
| SCAN t-i     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epoches=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=128 `|
| SCAN i-t     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=i2t --lambda_softmax=4 --num_epoches=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=128 `|

For MS-COCO:
| Method    | Arguments |
| :-------: | :-------: |
| SCAN t-i     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epoches=30 --lr_update=15 --learning_rate=.0005 --embed_size=1024 --batch_size=128 `|
| SCAN i-t     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=i2t --lambda_softmax=4 --num_epoches=30 --lr_update=15 --learning_rate=.0005 --embed_size=1024 --batch_size=128 `|

For Tencent-News:
| Method    | Arguments |
| :-------: | :-------: |
| SCAN t-i     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epoches=30 --lr_update=15 --learning_rate=.0002 --embed_size=512 --batch_size=128 `|

## Evaluate trained models

```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/f30k_precomp/model_best.pth.tar", data_path="$DATA_PATH", split="test")
