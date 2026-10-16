# All You Need

### From Tensor Mathematics to Language Models and Dynamic Memory Management

**All You Need** is an exercise-driven learning collection that connects the mathematics of neural networks with their implementation and the systems needed to run them. It follows a path through tensor operations, differentiation, optimization, data storage, neural network architectures, Transformers, large language models, and GPU memory engineering.

The central question is simple: **how do these systems actually work, and how can you build, inspect, and improve them yourself?**

The curriculum brings together **19 learning tracks, 184 chapters, and 3,189 exercises**. These counts come from the supplied exercise catalogue; they describe the curriculum's scope, rather than a claim that every implementation has been independently tested.

## What You Will Learn

Working through the exercises will help you develop the ability to:

- Translate mathematical operations into tensor code and reason about shapes, dimensions, and numerical behavior.
- Explain how gradients move through a computation graph and verify backpropagation with manual calculations.
- Implement optimization algorithms, loss functions, and complete training loops.
- Organize structured records, conversation history, model files, and embeddings using different storage systems.
- Build and compare regression models, perceptrons, multilayer networks, CNNs, RNNs, and LSTMs.
- Implement attention mechanisms and Transformer components, then connect them to language modeling and text generation.
- Investigate pretraining, fine-tuning, preference alignment, quantization, evaluation, and retrieval-augmented systems.
- Profile execution, diagnose numerical and memory problems, and measure the effect of your changes.
- Design a memory runtime that coordinates model state across GPU memory, CPU memory, and NVMe storage.

## Curriculum at a Glance

| # | Learning track | Chapter codes | Chapters | Exercises |
| --- | --- | --- | ---: | ---: |
| 1 | Tensor Mathematics and Neural Networks Guide | `T00–T08` | 9 | 139 |
| 2 | Applied Linear Algebra in Code | `ALA00–ALA06` | 7 | 60 |
| 3 | Autograd & Backprop Mechanics | `ABM00–ABM06` | 7 | 70 |
| 4 | Optimization in Code | `OC00–OC06` | 7 | 70 |
| 5 | Loss Functions & Training Loop | `LFTL00–LFTL06` | 7 | 70 |
| 6 | SQL Lessons — PostgreSQL | `SQL00–SQL03` | 4 | 40 |
| 7 | NoSQL Lessons — MongoDB | `NOSQL00–NOSQL03` | 4 | 40 |
| 8 | Object Storage Lessons — MinIO | `OB00–OB03` | 4 | 40 |
| 9 | Vector Database Lessons — Qdrant | `QD00–QD03` | 4 | 40 |
| 10 | Linear Regression | `LR00–LR08` | 9 | 180 |
| 11 | Logistic Regression | `LG00–LG08` | 9 | 180 |
| 12 | Perceptron | `PC00–PC08` | 9 | 180 |
| 13 | Multilayer Perceptron | `MLP00–MLP08` | 9 | 180 |
| 14 | Convolutional Neural Networks | `CNN00–CNN08` | 9 | 180 |
| 15 | Recurrent Neural Networks | `RNN00–RNN08` | 9 | 180 |
| 16 | Long Short-Term Memory | `LSTM00–LSTM08` | 9 | 180 |
| 17 | Transformers | `TRF00–TRF08` | 9 | 180 |
| 18 | Large Language Models | `LLM00–LLM08` | 9 | 180 |
| 19 | Dynamic Memory Management | `DMM00–DMM49` | 50 | 1,000 |
| | **Total** | | **184** | **3,189** |

## 1. Understand the Mathematics Behind the Code

The tensor mathematics and applied linear algebra tracks build the vocabulary needed to understand neural network computations.

You will study tensor arithmetic, comparisons, vector operations, matrix multiplication, higher-dimensional operations, and shape transformations. You will also encounter differential calculus, optimization, statistics, probability, and transformations used in neural networks.

The applied exercises connect these ideas to code: creating tensors, computing norms and dot products, distinguishing inner and outer products, applying broadcasting rules, and tracking dimensions through a sequence of operations. Basis, span, and projection introduce the geometry behind these calculations.

**Practical outcome:** you can inspect a tensor expression, explain its mathematical meaning, predict its output shape, and investigate incorrect results.

## 2. Understand How Neural Networks Learn

The autograd, optimization, and training-loop tracks explain the mechanics of learning.

You will construct computation graphs, distinguish forward and backward passes, calculate gradients manually, apply the chain rule, and inspect PyTorch's automatic differentiation behavior. Exercises on gradient flow connect the mathematics to debugging real implementations.

You will implement gradient descent, momentum, RMSProp, and Adam, then investigate learning rates, initialization, convergence, and stability. The loss-function track covers mean squared error, cross-entropy, batching, epochs, validation splits, and overfitting prevention.

**Practical outcome:** you can build a training loop, explain how a parameter update is calculated, and diagnose common reasons a model fails to learn.

## 3. Build the Data Layer Around AI Systems

Four storage tracks address different kinds of data used around models.

| Technology | What you will learn |
| --- | --- |
| **PostgreSQL** | Relational data, tables, keys, schemas, user and plan records, query logic, and transactions. |
| **MongoDB** | Document structures, chat history, session memory, flexible schemas, indexing, replication, and sharding. |
| **MinIO** | Object storage for datasets, model weights, and checkpoints; buckets, access policies, distributed operation, and fault tolerance. |
| **Qdrant** | Embeddings, similarity search, text retrieval workflows, retrieval-augmented generation, and scalable indexing. |

**Practical outcome:** you can distinguish the storage needs of application records, conversations, large model artifacts, and semantic search, and implement exercises for each.

## 4. Build Models from Regression to Multilayer Networks

Linear regression develops the relationship between inputs, predictions, error, gradients, and parameter updates. Its exercises also cover data preparation, convergence, evaluation, and visualization.

Logistic regression introduces binary classification, sigmoid behavior, probability interpretation, decision boundaries, classification loss, and multiclass extensions.

The perceptron track explores the learning rule, convergence, classification errors, multiclass variants, and the limits of linear decision boundaries. It also examines computational performance and when GPU execution is useful.

The multilayer perceptron track extends these ideas to networks with hidden layers and nonlinear activations. You will work on architecture, backpropagation, initialization, training stability, logging, regularization, dropout, profiling, and scaling.

**Practical outcome:** you can implement baseline models, explain their assumptions and limitations, and investigate when a more expressive architecture is needed.

## 5. Learn How Networks Process Images and Sequences

### Convolutional Neural Networks

You will explore convolution, filters, feature maps, receptive fields, pooling, downsampling, and invariance. Exercises extend into architecture implementation, training, image augmentation, normalization, regularization, robustness, modern convolutional blocks, and export.

**Practical outcome:** you can construct an image-processing model and investigate how its architecture and data pipeline affect learning and execution.

### Recurrent Neural Networks

You will study hidden states, temporal dependencies, backpropagation through time, sequence batching, language modeling, and time-series forecasting. Training stability and performance exercises connect recurrent behavior to practical limitations.

**Practical outcome:** you can implement a recurrent sequence model and explain why long dependencies and gradient behavior make training difficult.

### Long Short-Term Memory

You will examine why gates are introduced, how LSTM state evolves, and how architecture variants handle sequences. Exercises cover backpropagation, text generation, forecasting, regularization, profiling, and deployment considerations.

**Practical outcome:** you can compare vanilla RNNs with gated recurrent models and explain the motivation for moving toward attention.

## 6. Build an Understanding of Transformers

The Transformer track connects the motivation for attention to the components of a working architecture:

- Attention and the limitations of sequential processing.
- Multi-head attention and query, key, and value projections.
- Positional encoding and sequence order.
- Residual connections, layer normalization, and feed-forward blocks.
- Tokenization, embeddings, and vocabulary preparation.
- Training objectives, optimization, and stability.
- Decoder-only language modeling and generation.
- Encoder-decoder architectures, cross-attention, and sequence-to-sequence tasks.
- Efficiency, long-context behavior, and execution profiling.

**Practical outcome:** you can follow the path from tokens to model outputs and implement and inspect the major components used in Transformer language models.

## 7. Connect Model Architecture to LLM Workflows

The LLM track extends the architectural foundation into the model development and application lifecycle.

You will investigate tokenization and context, pretraining objectives and data, instruction tuning, fine-tuning, preference alignment, inference, KV caching, serving, quantization, and deployment. Evaluation exercises address hallucination and reliability, while the final chapter connects language models to tools, agents, and retrieval-augmented generation.

**Practical outcome:** you can explain how training, inference, retrieval, and tool use fit together, and build experiments that examine their behavior and tradeoffs.

## 8. Engineer Memory-Constrained Model Execution

Dynamic Memory Management is the largest track: **50 chapters and 1,000 exercises** focused on PyTorch, CUDA, and the movement and lifetime of model data.

| Area | What you will investigate |
| --- | --- |
| **Memory measurement** | CUDA memory APIs, allocated and reserved memory, allocator behavior, statistics, snapshots, and profiling. |
| **Asynchronous execution** | Pinned host buffers, non-blocking transfers, CUDA streams and events, prefetching, and compute-transfer overlap. |
| **Placement and offloading** | Tensor, parameter, and layer placement; CPU offloading; NVMe storage; and out-of-core execution. |
| **Training memory** | Activation checkpointing, recomputation, activation offloading, mixed precision, gradient scaling, accumulation, and adaptive batch sizing. |
| **Allocation and lifetime** | Memory pools, buffer reuse, fragmentation, in-place operations, storage aliasing, meta-device construction, and lazy materialization. |
| **Data access** | Memory-mapped datasets and DataLoader memory behavior. |
| **Inference memory** | Paged storage, paged KV caches, cache offloading, quantization, eviction, and context-retention policies. |
| **Multiple GPUs** | FSDP, ZeRO-style partitioning, tensor parallel placement, pipeline scheduling, and memory balancing. |
| **Diagnostics and recovery** | Bandwidth measurement, memory leaks, autograd graph retention, gradient tracking, and out-of-memory recovery. |
| **Runtime design** | Registries, memory budgets, residency states, eviction policies, cost models, schedulers, telemetry, and stress testing. |

The final chapters bring these mechanisms together into a custom memory manager and a three-tier offload runtime spanning **VRAM, RAM, and NVMe**. The exercises target models whose parameter footprint exceeds available VRAM, with attention to correctness, transfer costs, scheduling, and failure recovery.

**Practical outcome:** you can reason about where model state lives, why execution runs out of memory, and how to design and benchmark policies that trade capacity, computation, and transfer time.

## Projects the Curriculum Builds Toward

The exercise sequence provides a foundation for projects such as:

- A manually implemented training pipeline with gradient checks and optimizer comparisons.
- Regression and classification experiments with evaluation and diagnostic plots.
- An image classifier with augmentation and performance measurements.
- Recurrent and LSTM models for text generation or time-series forecasting.
- A small Transformer language model with tokenization and generation.
- A retrieval workflow combining embeddings and vector search.
- A GPU memory diagnostic workflow for identifying leaks, retained graphs, and transfer bottlenecks.
- A custom PyTorch memory manager and offload engine with explicit memory budgets and runtime policies.

These are learning targets. Model size, achievable speed, and experiment scale depend on the implementation, data, and hardware available.

## How to Study

Start with the mathematical and training foundations, then move through the model tracks toward Transformers and LLMs. Study the storage tracks when building data-backed applications. Approach the advanced memory track after becoming comfortable with PyTorch training and GPU execution.

For each exercise:

1. Explain the concept and predict the expected behavior.
2. Implement the calculation or experiment.
3. Check correctness against a manual result, reference implementation, or controlled baseline.
4. Change one variable and observe what happens.
5. Record the result, failure cases, and the explanation you can support with evidence.

For performance work, measure both correctness and resource use. A faster result is useful only when it preserves the behavior required by the task.

## Background and Tools

Basic Python programming and elementary algebra are useful starting points. Later exercises assume growing familiarity with tensors, differentiation, PyTorch, and command-line tools.

The curriculum includes **PyTorch and CUDA**, alongside **PostgreSQL, MongoDB, MinIO, and Qdrant**. CPU execution can support many foundational experiments; CUDA-specific and multi-GPU exercises require suitable hardware. NVMe offloading experiments also depend on available storage capacity and bandwidth.

The goal is to develop the habit of connecting **mathematics, implementation, experimental evidence, and system behavior** throughout the learning process.
