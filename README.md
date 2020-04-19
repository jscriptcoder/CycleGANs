# Cycle Generative Adversarial Networks

This project is based on the paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN), and part of [Udacity Deep Learning Nanodegree program](https://www.udacity.com/course/deep-learning-nanodegree--nd101)

The goal is to train generators that learn to transform an image from domain **X** into an image that looks like it came from domain **Y** (and vice versa).

Examples:

<img src="assets/cycle_gan_examples.jpeg" />

## Unpaired Training Data

These images do not come with labels, but CycleGANs give us a way to learn the mapping between one image domain and another using an **unsupervised** approach. A CycleGAN is designed for image-to-image translation and it learns from unpaired training data. This means that in order to train a generator to translate images from domain **X** to domain **Y**, we do not have to have exact correspondences between individual images in those domains. For example, in the [paper that introduced CycleGANs](https://arxiv.org/abs/1703.10593), the authors are able to translate between images of horses and zebras, even though there are no images of a zebra in exactly the same position as a horse or with exactly the same background, etc. Thus, CycleGANs enable learning a mapping from one domain **X** to another domain **Y** without having to find perfectly-matched, training pairs!

<img src="assets/horse2zebra.gif" width=50% />

## Architecture

A CycleGAN is made of two types of networks: **discriminators**, and **generators**. The discriminators are responsible for classifying images as real or fake (for both **X** and **Y** kinds of images). The generators are responsible for generating convincing, fake images for both kinds of images.

### Discriminator
TODO

### Generator
TODO
