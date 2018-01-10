Parametric Exponential Linear Unit (PELU) for ResNet training in Torch
============================

This project is a clone of [Facebook ResNet implementation using ReLU](https://github.com/facebook/fb.resnet.torch).

This implements training of residual networks from [Parametric Exponential Linear Unit for Deep Convolutional Neural Networks](http://arxiv.org/abs/1605.09332) by Trottier, L., et. al (2016).



## Requirements

See the [installation instructions](INSTALL.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU
- Install [cuDNN v4](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4) 

If you already have Torch installed, update `nn`, `cunn`, and `cudnn`.



## Training

The training commands are available in the following scripts:

* PELU:  `experiment.pelu.sh` .
* ELU: `experiment.elu.sh`
* ReLU: `experiment.bnrelu.sh`
* PReLU: `experiment.pelu.sh`
* BN+PELU: `experiment.bnpelu.sh`
* BN+ELU: `experiment.bnelu.sh`



## Results

Run `python show_results.py`  to get the following results:

```
results/results-pelu-div-mul - cifar10: mean [5.5076000000000001], min [5.3609999999999998]
results/results-pelu-div-mul - cifar100: mean [25.0212], min [24.550999999999998]
results/results-bn-pelu-div-mul - cifar10: mean [6.2442000000000011], min [5.8499999999999996]
results/results-bn-pelu-div-mul - cifar100: mean [26.044799999999999], min [25.381]
results/results-elu - cifar10: mean [6.5468000000000002], min [5.9859999999999998]
results/results-elu - cifar100: mean [26.589600000000001], min [25.077999999999999]
results/results-bn-elu - cifar10: mean [11.195399999999998], min [10.391]
results/results-bn-elu - cifar100: mean [35.517600000000002], min [34.746000000000002]
results/results-bn-relu - cifar10: mean [5.6738], min [5.4100000000000001]
results/results-bn-relu - cifar100: mean [25.919799999999999], min [24.989999999999998]
results/results-bn-prelu - cifar10: mean [5.6054000000000004], min [5.3609999999999998]
results/results-bn-prelu - cifar100: mean [25.8262], min [25.498000000000001]
results/results-pelu-div-div - cifar10: mean [5.7306000000000008], min [5.5960000000000001]
results/results-pelu-div-div - cifar100: mean [25.683600000000002], min [25.166]
results/results-pelu-mul-div - cifar10: mean [6.5135999999999994], min [6.0060000000000002]
results/results-pelu-mul-div - cifar100: mean [26.3322], min [25.478999999999999]
results/results-pelu-mul-mul - cifar10: mean [6.7362000000000011], min [6.1230000000000002]
results/results-pelu-mul-mul - cifar100: mean [26.2012], min [25.244]
```




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
