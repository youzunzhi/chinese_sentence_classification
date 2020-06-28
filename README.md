# CNN for Chinese Sentence Classification

## Original Paper

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

## Getting Started

### Installation

```
git clone https://github.com/youzunzhi/chinese_sentence_classification.git
```

### Prerequisites
```
python 3.x 
PyTorch >= 1.0.1 
yacs
torchtext
```

### Download Pretrained word2vec

[https://pan.baidu.com/s/1VGOs0RH7DXE5vRrtw6boQA](https://pan.baidu.com/s/1VGOs0RH7DXE5vRrtw6boQA)

（from [https://github.com/Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)）

## Training

You can modify the training configs in `cfg` in the code or in command line following the usage of yacs. To quickly training with the four model variations, you can change the `EXPERIMENT_NAME` to one of `[baseline|pretrain|pretrain_finetune|multichannel]`, and the corresponding configs will be changed automatically. The `DATASET_NAME` choices are `[flight|laptop|movie]`

```shell
python train.py EXPERIMENT_NAME baseline DATASET_NAME flight
```