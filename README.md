# RainTRN: A Spatio-Temporal Transformer Network with Spectral Normalization for Video De-raining

**Authors:** Debayan Dutta, CV Team VIT

This repository contains the official PyTorch implementation for the paper: **"RainTRN: A Spatio-Temporal Transformer Network with Spectral Normalization for Video De-raining"**.

**RainTRN** is a novel deep learning architecture for robust video de-raining. It leverages a recurrent pipeline where the restored output from the preceding frame is used to maintain temporal consistency in the current frame. The model introduces two primary innovations:

1. A **Spatio-Temporal Transformer Network (STTN)** is used not for motion compensation, but as a powerful attention-based module to refine deep feature representations.

2. **Spectral Normalization (SN)** is incorporated within the discriminator of an adversarial training framework to regularize and stabilize the training process, leading to higher-quality perceptual results.

## Table of Contents

* [Key Contributions](#key-contributions)
* [Model Architecture](#model-architecture)
* [License](#license)

## Key Contributions

* **Novel Recurrent Architecture:** A new video de-raining model that combines a recurrent pipeline with a transformer-based module for sophisticated, attention-based feature refinement.

* **Stabilized Adversarial Training:** The first model in this domain to leverage Spectral Normalization (SN) within its GAN framework to ensure training stability and generate perceptually superior results.

* **New Real-World Benchmark Dataset:** We introduce a new, large-scale dataset of paired real-world rainy videos and their corresponding clean ground truths, captured across diverse scenes and rain conditions, to facilitate more realistic training and evaluation.

## Model Architecture

RainTRN is designed as a recurrent pipeline that processes video frame-by-frame, leveraging temporal information from the previously restored frame to de-rain the current one. The entire de-raining network functions as the generator within a Generative Adversarial Network (GAN) framework for training.

### Problem Formulation

The formation of a rainy frame `y_t` is modeled as a linear superposition of the clean background `x_t` and a rain layer `r_t`.

```
y_t = x_t + r_t
```

The objective of RainTRN is to learn a mapping function `F` that recovers the clean frame `x_t` by processing the video recurrently:

```
x̂_t = F(y_t, x̂_{t-1})
```

Here, `x̂_t` is the estimated clean frame at time `t`, and the initial state `x̂_0` is a zero tensor.

### Architectural Pipeline

For each time step `t`, the model takes the current rainy frame `y_t` and the previously restored frame `x̂_{t-1}` as input. The data flows through the network as follows:

1. **Input & Feature Extraction:** The current rainy frame is concatenated with the previous de-rained output along the channel dimension. This combined tensor is then passed through an initial convolutional layer followed by a series of residual blocks to encode the input into a high-dimensional feature map.

2. **Transformer-based Feature Refinement:** The extracted feature map is processed by the Spatio-Temporal Transformer Network (STTN) module. Unlike traditional approaches that use transformers for explicit motion compensation, the STTN here acts as a powerful feature refinement block.

3. **Decoding & Reconstruction:** The refined features from the STTN are added back to the pre-STTN features via a residual skip connection. This combined feature map is then passed through a final set of convolutional layers to decode the features and reconstruct the final de-rained output frame `x̂_t`.

### Spatio-Temporal Transformer Network (STTN) Core

The STTN is the heart of RainTRN's feature processing capability. It is composed of a stack of TransformerBlocks, where each block contains two main sub-modules:

* **Multi-Head Attention:** The input feature map is first divided into non-overlapping patches. Scaled dot-product attention is then computed between these patches, allowing the model to learn complex spatial relationships and selectively focus on the most informative features across the entire map. This mechanism refines the feature representation without explicitly calculating motion or warping frames.

* **Feed-Forward Network:** After the attention module, a position-wise feed-forward network, implemented as a pair of convolutional layers, is applied to further process each feature within its spatial context.

Each TransformerBlock uses residual connections around both the attention and feed-forward modules to facilitate deep network training.

### Spectral Normalization in Adversarial Training

To generate perceptually high-quality and realistic results, the RainTRN model is trained as a generator within a GAN setup.

* **Discriminator:** A separate 3D convolutional discriminator network is trained to distinguish between sequences of real clean frames and the de-rained frames produced by RainTRN.

* **Stabilization:** To ensure this adversarial training process is stable, **Spectral Normalization (SN)** is applied to the weight matrices of the convolutional layers within the discriminator network. SN regularizes the discriminator by constraining its Lipschitz constant, which prevents its gradients from becoming too large or vanishing and helps to balance the training between the generator and discriminator. SN is not applied to the main de-raining generator network.


## License

All rights reserved.
