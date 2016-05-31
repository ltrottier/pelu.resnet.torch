Parametric Exponential Linear Unit (PELU) for ResNet training in Torch
============================

This project is a clone of [Facebook ResNet implementation using ReLU](https://github.com/facebook/fb.resnet.torch).

This implements training of residual networks from [Parametric Exponential Linear Unit for Deep Convolutional Neural Networks](http://arxiv.org/abs/1512.03385) by Trottier, L., et. al (2016).

## Requirements
See the [installation instructions](INSTALL.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU
- Install [cuDNN v4](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)
- (Optional) Download the [ImageNet](http://image-net.org/download-images) dataset and [move validation images](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) to labeled subfolders

If you already have Torch installed, update `nn`, `cunn`, and `cudnn`.

## Training

(See the [training recipes](TRAINING.md) for additional examples.)

To get the same results as [Parametric Exponential Linear Unit for Deep Convolutional Neural Networks](http://arxiv.org/abs/1512.03385), use the following commands.

### CIFAR-10
```bash
th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -nEpochs 200 -depth 110 -shortcutType A -weightDecay 0.001 
```

You should get around 5.4% top 1 error.

### CIFAR-100
```bash
th main.lua -dataset cifar100 -nGPU 2 -batchSize 128 -nEpochs 200 -depth 110 -shortcutType A -weightDecay 0.001
```

You should get around 25.5% top 1 error.


## ResNet ReLU vs ResNet ELU/PELU

There are 3 differences between [Facebook ResNet implementation using ReLU](https://github.com/facebook/fb.resnet.torch) and this implementation using PELU:

1. We remove Batch Normalization (BN) before the activation function. This was pointed out by [Shah, A., et. al. (2016)](https://arxiv.org/pdf/1604.04112.pdf), for ELU, where using BN degraded performances.
2. We remove the activation function after the skip connection. Again pointed out [Shah, A., et. al. (2016)](https://arxiv.org/pdf/1604.04112.pdf), for ELU.
3. We added an activation function before the last average pooling.





## Notes

This implementation differs from the ResNet paper in a few ways:

**Scale augmentation**: We use the [scale and aspect ratio augmentation](datasets/transforms.lua#L130) from [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842), instead of [scale augmentation](datasets/transforms.lua#L113) used in the ResNet paper. We find this gives a better validation error.

**Color augmentation**: We use the photometric distortions from [Andrew Howard](http://arxiv.org/abs/1312.5402) in addition to the [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)-style color augmentation used in the ResNet paper.

**Weight decay**: We apply weight decay to all weights and biases instead of just the weights of the convolution layers.

**Strided convolution**: When using the bottleneck architecture, we use stride 2 in the 3x3 convolution, instead of the first 1x1 convolution.
