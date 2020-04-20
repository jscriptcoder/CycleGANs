# Cycle-Consistent Adversarial Networks

WORK IN PROGRESS!!

This project is based on the paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN), and part of [Udacity Deep Learning Nanodegree program](https://www.udacity.com/course/deep-learning-nanodegree--nd101)

The goal is to train generators that learn to transform an image from domain **X** into an image that looks like it came from domain **Y** (and vice versa).

Examples:

<img src="assets/cycle_gan_examples.jpeg" width="100%" />

## Unpaired Training Data

These images do not come with labels, but CycleGANs give us a way to learn the mapping between one image domain and another using an **unsupervised** approach. A CycleGAN is designed for image-to-image translation and it learns from unpaired training data. This means that in order to train a generator to translate images from domain **X** to domain **Y**, we do not have to have exact correspondences between individual images in those domains. For example, in the [paper that introduced CycleGANs](https://arxiv.org/abs/1703.10593), the authors are able to translate between images of horses and zebras, even though there are no images of a zebra in exactly the same position as a horse or with exactly the same background, etc. Thus, CycleGANs enable learning a mapping from one domain **X** to another domain **Y** without having to find perfectly-matched, training pairs!

<img src="assets/horse2zebra.gif" width=50% />

## Architecture

A CycleGAN is made of two types of networks: **discriminators**, and **generators**. The discriminators are responsible for classifying images as real or fake (for both **X** and **Y** kinds of images). The generators are responsible for generating convincing, fake images for both kinds of images.

### Discriminator

The discriminators, `D_X` and `D_Y`, are convolutional neural networks that see an image and attempt to classify it as real or fake. In this case, real is indicated by an output close to 1 and fake as close to 0. The discriminators have the following architecture:

<img src="assets/discriminator_layers.png" width="100%" />

This network sees an image, and passes it through 5 convolutional layers that downsample the image by a factor of 2. The first four convolutional layers have a BatchNorm and ReLu activation function applied to their output, and the last acts as a classification layer that outputs one value.

### Generator
The generators, `G_XtoY` and `G_YtoX` (sometimes called **G** and **F**), are made of an **encoder**, a *conv* net that is responsible for turning an image into a smaller feature representation, and a **decoder**, a *transpose_conv* net that is responsible for turning that representation into an transformed image. These generators, one from XtoY and one from YtoX, have the following architecture:

<img src="assets/cyclegan_generator_ex.png" width="100%" />

This network sees an image, compresses it into a feature representation as it goes through three convolutional layers and reaches a series of residual blocks. It goes through a few (typically 6 or more) of these residual blocks, then it goes through three transpose convolutional layers (sometimes called *de-conv* layers) which upsample the output of the resnet blocks and create a new image!

Note that most of the convolutional and transpose-convolutional layers have BatchNorm and ReLu functions applied to their outputs with the exception of the final transpose convolutional layer, which has a `tanh` activation function applied to the output. Also, the residual blocks are made of convolutional and batch normalization layers.

### Residual Blocks
The motivation for using residual blocks is as follows: very deep neural networks can be difficult to train. Deeper networks are more likely to have vanishing or exploding gradients and, therefore, have trouble reaching convergence; batch normalization helps with this a bit. However, during training, we often see that deep networks respond with a kind of training degradation. Essentially, the training accuracy stops improving and gets saturated at some point during training. In the worst cases, deep models would see their training accuracy actually worsen over time!

One solution to this problem is to use **Resnet blocks** that allow us to learn so-called residual functions as they are applied to layer inputs. You can read more about this proposed architecture in the paper, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He et. al, and the below image is from that paper.

<p align="center"><img src="assets/resnet_block.png" width="50%" /></p>

#### Residual Functions

Usually, when we create a deep learning model, the model (several layers with activations applied) is responsible for learning a mapping `M` from an input `x` to an output `y`

`M(x) = y` 

Instead of learning a direct mapping from `x` to `y`, we can instead define a residual function

`F(x) = M(x) - x`

This looks at the difference between a mapping applied to `x` and the original input, `x`. `F(x)` is, typically, two convolutional layers + normalization layer and a ReLu in between. These convolutional layers should have the same number of inputs as outputs. This mapping can then be written as the following; a function of the residual function and the input `x`. The addition step creates a kind of loop that connects the input `x` to the output `y`:

`M(x) = F(x) + x`  or `y = F(x) + x`

#### Optimizing a Residual Function

The idea is that it is easier to optimize this residual function `F(x)` than it is to optimize the original mapping `M(x)`. Consider an example; what if we want `y = x`? From our first direct mapping equation, `M(x) = y` , we could set `M(x) = x` but it is easier to solve the residual equation `F(x) = 0`, which, when plugged in to `y = F(x) + x`, yields `y = x`.
