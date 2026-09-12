# Deep Learning Problem Set

This repository contains my work for the **Deep Learning** course.

The project is organized into six problem sets covering different areas of
deep learning, from convolutional and recurrent networks to generative models
and Transformers.

My main goal throughout these experiments was not only to train the models,
but also to understand how architectural choices affect accuracy, training
behavior, robustness, representation, and generation.

## Project Overview

| Problem | Topic | Main Experiments |
|---|---|---|
| 1 | CNNs & Robustness | ResNet-18/34/50, CIFAR-10/100, Linear Mode Connectivity, FGSM, Adversarial Training |
| 2 | Sequence Modeling | RNN, LSTM, GRU sentiment classification, LSTM vs. Transformer forecasting |
| 3 | Autoencoders & VAEs | SVD, Linear/Non-linear Autoencoders, latent-space analysis, VAE architectures |
| 4 | GANs | DCGAN, WGAN-GP, FID evaluation and training stability |
| 5 | Transformers | Character-level GPT, attention analysis, context length, spatial locality |
| 6 | Diffusion Models | Unconditional DDPM, Conditional DDPM, Classifier-Free Guidance |

---

## Problem 1 — CNN Architecture and Robustness

The first problem explores how network depth and architectural choices affect
image classification and robustness.

I implemented **ResNet-18, ResNet-34, and ResNet-50** and trained them on
CIFAR-10 and CIFAR-100. I also studied Linear Mode Connectivity and the effect
of Batch Normalization and residual connections.

The final part focuses on adversarial robustness using the **FGSM attack** and
adversarial training.

### Selected results

| Model | CIFAR-10 Accuracy | CIFAR-100 Accuracy |
|---|---:|---:|
| ResNet-18 | 87.35% | 60.64% |
| ResNet-34 | **87.85%** | 59.84% |
| ResNet-50 | 85.71% | 53.86% |

For CIFAR-10, ResNet-34 gave the best balance between accuracy and model
complexity. Increasing the depth to ResNet-50 did not improve the result.

The robustness experiment showed an even clearer difference:

| ResNet-34 | Clean Accuracy | FGSM Robust Accuracy |
|---|---:|---:|
| Standard training | 87.85% | 33.58% |
| Adversarial training | 82.77% | **66.09%** |

Adversarial training improved robust accuracy by **32.51 percentage points**,
at the cost of a small reduction in clean accuracy.

---

## Problem 2 — Sequence Modeling

This problem studies sequential data in two different settings: natural
language and time-series forecasting.

For sentiment classification, I trained **RNN, LSTM, and GRU** models on the
SST-2 dataset using pre-trained GloVe embeddings.

### Sentiment classification

| Architecture | Final Validation Accuracy |
|---|---:|
| RNN | 81.54% |
| LSTM | 85.09% |
| GRU | **85.21%** |

Both LSTM and GRU clearly improved over the basic RNN, with GRU producing the
best final validation accuracy.

The second experiment compares an **LSTM and a Transformer** for CO₂
time-series forecasting using different lookback windows and forecast
horizons.

The LSTM achieved a lower RMSE in every tested configuration. The best result
was obtained with a lookback of 52 and a forecast horizon of 1:

- LSTM RMSE: **6.197**
- Transformer RMSE: **10.629**

This experiment was a useful example of why a Transformer is not automatically
the best choice for every sequence problem, especially when the dataset is
relatively small and strongly structured.

---

## Problem 3 — Autoencoders and Variational Autoencoders

In this problem, I explored dimensionality reduction, reconstruction, latent
representations, and generative modeling.

The first experiment compares **SVD, a Linear Autoencoder, and a Non-linear
Autoencoder** on CIFAR-10 using the same latent dimension.

| Method | Test MSE |
|---|---:|
| SVD | **0.008719** |
| Linear Autoencoder | 0.021000 |
| Non-linear Autoencoder | 0.018087 |

SVD achieved the lowest reconstruction error, while the Non-linear Autoencoder
produced somewhat more structured latent representations in the t-SNE
visualizations.

The second part uses Fashion-MNIST to compare three VAE architectures:
**Base, Deeper, and Wider**.

The Wider VAE gave the best overall balance between reconstruction quality and
generation, with a reconstruction loss of about **16.63**.

I also compared the Wider VAE with a standard Autoencoder. The standard
Autoencoder reconstructed images more accurately, but random points in its
latent space produced poor samples. The VAE produced a much smoother and more
useful latent space for generation and interpolation.

---

## Problem 4 — Generative Adversarial Networks

This problem compares two GAN training approaches on CIFAR-10:

- **DCGAN**
- **WGAN-GP**

I followed their training behavior, generated samples during training, and
used **Fréchet Inception Distance (FID)** to evaluate sample quality.

| Model | Final FID |
|---|---:|
| DCGAN | **68.65** |
| WGAN-GP | 164.11 |

In this experiment, DCGAN produced the lower final FID.

This result was also a useful reminder that a theoretically more stable
training method does not necessarily produce a better final result under every
training budget and hyperparameter configuration.

---

## Problem 5 — Understanding Transformers

For this problem, I wanted to look beyond simply training a Transformer and
inspect what happens inside it.

I implemented a small GPT-style character-level language model using the
**Tiny Shakespeare** dataset. The model contains:

- 4 Transformer layers
- 4 attention heads
- 128-dimensional embeddings

I analyzed attention maps and measured properties such as attention distance,
entropy, self-attention, and attention to previous tokens.

I also experimented with context lengths of **32, 64, and 128**.

| Context Length | Validation Loss | Perplexity | Peak GPU Memory |
|---:|---:|---:|---:|
| 32 | **2.481** | **11.95** | 81.6 MB |
| 64 | 2.510 | 12.31 | 133.5 MB |
| 128 | 2.515 | 12.37 | 261.9 MB |

Under the same short training budget, increasing the context length increased
the computational cost without improving validation performance.

For the final experiment, I built a separate Patch Transformer on the Digits
dataset to study spatial locality in deeper Transformer layers.

The model reached about **95.56% test accuracy**.

The experiments showed that spatial information gradually becomes mixed as
representations move through deeper layers, but it is not completely lost.
Residual connections, patch content, and positional information all contribute
to this behavior.

---

## Problem 6 — Denoising Diffusion Probabilistic Models

The final problem focuses on **DDPMs** and the process of generating images by
gradually reversing noise.

I implemented both:

- an Unconditional DDPM
- a Conditional DDPM with Classifier-Free Guidance

Both models were trained on CIFAR-10 using a compact U-Net.

### FID comparison

| Model | FID |
|---|---:|
| Unconditional DDPM | 292.88 |
| Conditional DDPM + CFG | **263.60** |

Adding class information and Classifier-Free Guidance improved the FID score
by about 29 points in this experiment.

These DDPM experiments were intentionally kept small because of computational
limits, and the FID values were calculated using only 64 generated samples.
For that reason, the absolute FID values should be interpreted carefully.

---

## What I Took Away From This Project

One of the most useful things I learned from these experiments is that a larger
or more complex model is not automatically a better model.

ResNet-50 did not outperform the smaller ResNets, the LSTM performed better
than the Transformer on the CO₂ forecasting task, a standard Autoencoder gave
better reconstruction but a worse generative latent space than the VAE, and a
longer Transformer context increased memory usage without improving the result
under the same training budget.

Working through these differences made the trade-offs between architecture,
optimization, computation, robustness, and generalization much more concrete
for me.

---

## Repository Structure

```text
deep-learning-problem-set/
│
├── problem-1-resnet-robustness/
├── problem-2-sequence-modeling/
├── problem-3-autoencoders-vaes/
├── problem-4-gans/
├── problem-5-transformers/
├── problem-6-diffusion-models/
│
├── report/
│
├── requirements.txt
└── README.md

```

Each problem folder contains the corresponding notebook and experiment code.
The report/ directory contains the complete project report with detailed code
explanations, experiment results, figures, and analysis.

Each problem folder contains the corresponding notebook and experiment code.

The report/ directory contains the complete project report with detailed code
explanations, experiment results, figures, and analysis.

---

## Running the Project
The experiments were developed and tested mainly in Google Colab with GPU
acceleration.

To install the required Python packages:
pip install -r requirements.txt

Some notebooks save results to Google Drive, so the output paths may need to be
changed before running them in a different environment.

---

## Tools and Libraries

Python, PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Scikit-learn,
Hugging Face Datasets, Gensim, Statsmodels, TorchMetrics, and Torch-Fidelity.

---

Author: Faeze Ahmadi
Deep Learning Problem Set
Institute for Advanced Studies in Basic Sciences (IASBS)
2026
