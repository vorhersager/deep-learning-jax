

---

# **Accelerated Architectures: Building Diffusion, Transformers, Language Models, and RL in JAX**

### **Foreword**

The transition from understanding deep learning in theory to deploying it in practice is rarely linear. Too often, the underlying mechanics of modern AI are obscured behind high-level APIs and black-box frameworks, leaving practitioners equipped to call functions but ill-prepared to engineer novel solutions.

Originally developed for graduate-level instruction in neural networks and deep learning, this collection of tutorials is designed to bridge that exact gap. The purpose of this repository is to provide stand-alone, transparent reference implementations of the most critical concepts in modern artificial intelligence. By leveraging the composability and raw performance of the JAX ecosystem, these notebooks peel back the layers of abstraction.

The curriculum follows a deliberate evolution. It begins at the absolute bedrock—computational linear algebra, information theory, and manual backpropagation—before scaling up to the sophisticated architectures dominating today’s landscape. By building systems like Generative Adversarial Networks, Diffusion models, and Decoder-only Transformers entirely from scratch, the mathematical intuition becomes concrete code.

These tutorials are not just academic exercises; they are forged to be starting points. Whether you are building a custom optimizer, fine-tuning a domain-specific large language model, or designing a reinforcement learning environment, these reference implementations are meant to be adapted, scaled, and deployed to solve complex, real-world engineering problems.

---

### **Tutorial Summaries**

**Tutorial 1: Mathematical Foundations in JAX and TensorFlow**

* **What it teaches:** Establishes the core computational groundwork by comparing TensorFlow's eager execution with JAX's composable transformations. It covers numerical stability in matrix inversion, automatic differentiation (Jacobians), and information theory measures like Cross-Entropy and KL Divergence.  
* **Real-World Application:** Essential for researchers and engineers who need to design custom loss functions, diagnose numerical instability in legacy systems, or migrate older operations to highly accelerated hardware (TPUs/GPUs).

**Tutorial 2: Linear Models, MLP, Backprop and Optimization**

* **What it teaches:** The transition from analytical math to iterative deep learning. It walks through solving linear and ridge regression analytically, followed by manually implementing the chain rule for backpropagation in a Multilayer Perceptron (MLP) and verifying it against JAX’s Autodiff.  
* **Real-World Application:** Foundational for deploying lightweight, highly interpretable models on edge devices or IoT sensors where importing a massive deep learning framework is computationally prohibitive.

**Tutorial 3: Nonconvex Optimization in Deep Learning**

* **What it teaches:** An exploration of how neural networks navigate complex loss landscapes. It implements key optimizers (SGD, Nesterov, RMSProp, Adam) from scratch and features 3D surface visualizations to illustrate how different algorithms escape saddle points and local minima.  
* **Real-World Application:** Crucial for machine learning engineers diagnosing training instability, vanishing gradients, or designing custom learning rate schedules for highly specialized, non-standard datasets.

**Tutorial 4: Convolutional Neural Networks and Transfer Learning**

* **What it teaches:** Using the Flax library to build and train CNNs. It focuses on the mechanics of transfer learning, demonstrating how to freeze a pre-trained feature-extraction backbone and only update the final classification layers for a new task.  
* **Real-World Application:** The standard approach for rapidly deploying computer vision systems—such as industrial defect inspection or medical image analysis—when labeled training data is scarce and expensive to acquire.

**Tutorial 5: RNNs and LSTMs on Time Series Prediction**

* **What it teaches:** Tackles multivariate time-series forecasting using Sequence-to-Sequence (Seq2Seq) modeling. It contrasts a vanilla Recurrent Neural Network built from scratch in pure JAX with a robust LSTM implementation using the Equinox library.  
* **Real-World Application:** Directly applicable to temporal sequence problems like predictive maintenance, algorithmic financial forecasting, or detecting anomalies in high-frequency flight telemetry data.

**Tutorial 6: Implementing a Toy Transformer from Scratch**

* **What it teaches:** A deep dive into the architecture powering modern AI. This tutorial builds a generative "Nano-GPT" block-by-block, covering character-level tokenization, Causal Self-Attention, and Multi-Layer Perceptrons, complete with dynamic visualizations of neuron activations.  
* **Real-World Application:** Understanding the exact mechanics of attention bottlenecks, which is necessary when modifying sequence models for non-text domains like genomics, protein folding, or specialized log-parsing.

**Tutorial 7: Generating Synthetic Faces using VAEs and Wasserstein GANs**

* **What it teaches:** The leap into Generative AI. It starts with Variational Autoencoders (VAEs) to learn continuous probability distributions, then extracts the VAE's decoder to use as a Generator in a Wasserstein GAN (WGAN) to synthesize highly realistic textures through adversarial training.  
* **Real-World Application:** Used extensively for generating synthetic, privacy-compliant training data, upscaling media, or creating baseline models for anomaly detection in manufacturing pipelines.

**Tutorial 8: Investigating Text-to-Image Transformation using Diffusion**

* **What it teaches:** The mathematics behind state-of-the-art image generation. It constructs a system using a Transformer-based text encoder and a Conditioned U-Net, relying heavily on the implementation of Classifier-Free Guidance (CFG) to sculpt images from pure static.  
* **Real-World Application:** The architectural foundation for building custom generative design tools for rendering, architecture, marketing, and automated synthetic content generation.

**Tutorial 9: Training a Self-Driving Racer with Reinforcement Learning**

* **What it teaches:** Off-policy deep reinforcement learning for continuous control. It builds a Soft Actor-Critic (SAC) agent from scratch using JAX/Optax and integrates DeepMind's Reverb for advanced memory techniques like Prioritized and Hindsight Experience Replay (PER/HER).  
* **Real-World Application:** The exact techniques used to train robotic control systems, optimize complex supply chain logistics, and develop dynamic pathing for autonomous vehicles.

**Tutorial 10: Large Language Models with GPT-2**

* **What it teaches:** The complete modern LLM lifecycle. Moving beyond the toy transformer, it implements Grouped-Query Attention (GQA) and walks through Unsupervised Pre-Training, Supervised Fine-Tuning (SFT), LoRA (Parameter-Efficient Fine-Tuning), RLHF, and distributed training via JAX pjit and FSDP.  
* **Real-World Application:** Provides the blueprint for enterprise teams needing to securely fine-tune domain-specific LLMs on proprietary data without relying on external, closed-source APIs.

**Tutorial 11: Model Explainability**

* **What it teaches:** This advanced tutorial delves into the mechanics of model interpretability and the "Clever Hans" phenomenon in AI—where a model learns a dataset's spurious correlations rather than true underlying concepts. Using JAX and Equinox, you will intentionally "poison" the CIFAR-10 dataset by adding an "AI" watermark to dogs, training a CNN to learn this shortcut. You will then act as an AI forensic investigator, utilizing four primary Explainable AI (XAI) lenses to catch the model cheating:  
* **Integrated Gradients (IG):** A gradient-based path integration method.  
* **Layer-wise Relevance Propagation (LRP):** A mechanistic backward score redistribution.  
* **LIME:** A model-agnostic behavioral probe using local surrogates.  
* **KernelSHAP:** A game-theoretic pixel attribution method.  
* **Real-World Application:** Extremely critical for AI safety, compliance, and debugging in high-stakes fields like healthcare, finance, or legal tech. It teaches engineers how to audit "black box" models, prove whether a system is making decisions based on legitimate biological/financial features or biased shortcuts (like watermarks, background artifacts, or protected attributes), and build trust with stakeholders before deployment.

