# Research Plan: PyTorch Deep Dive with BERT Fine-Tuning for Embedding-Based Retrieval

## Overview

This tutorial aims to provide a rigorous, hands-on introduction to PyTorch, culminating in fine-tuning a BERT model for embedding-based retrieval. It is designed for practitioners who want both theoretical grounding and practical experience — not just copy-paste code.

---

## Goals

- **In-depth understanding of PyTorch**: tensors, autograd, training loops, and model abstractions
- **Mathematical and conceptual foundation**: understand the theory behind dense embeddings and retrieval
- **Hands-on experience**: build and fine-tune a real BERT-based retrieval model end-to-end

---

## Module 1: PyTorch Fundamentals

### 1.1 Tensors and Operations
- What is a tensor? Relationship to numpy arrays
- Creating tensors: `torch.tensor`, `torch.zeros`, `torch.randn`
- Tensor operations: broadcasting, indexing, reshaping
- Device management: CPU vs. CUDA (`.to(device)`)

### 1.2 Autograd and Backpropagation
- Computational graphs: how PyTorch builds them dynamically
- `requires_grad`, `.backward()`, `.grad`
- The chain rule in practice: a manual example
- `torch.no_grad()` and when to use it

### 1.3 Building Neural Networks with `torch.nn`
- `nn.Module`: defining layers and `forward()`
- Common layers: `nn.Linear`, `nn.Dropout`, `nn.LayerNorm`
- Parameter management: `model.parameters()`, `named_parameters()`
- Saving and loading models: `state_dict`

### 1.4 Training Loop Anatomy
- `DataLoader` and `Dataset`: batching and shuffling
- Optimizers: `torch.optim.Adam`, learning rate scheduling
- The canonical training loop: zero_grad → forward → loss → backward → step
- Evaluation mode: `model.train()` vs. `model.eval()`

---

## Module 2: Embedding-Based Retrieval — Theory and Math

### 2.1 What is Embedding-Based Retrieval?
- Traditional retrieval (BM25, TF-IDF) vs. dense retrieval
- The core idea: map queries and documents into a shared vector space

### 2.2 Mathematical Foundations
- Vector spaces and inner product spaces
- Similarity metrics:
  - Dot product: $s(q, d) = \mathbf{q}^\top \mathbf{d}$
  - Cosine similarity: $s(q, d) = \frac{\mathbf{q}^\top \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$
- Why high-dimensional embeddings work: the geometry of semantic meaning

### 2.3 Contrastive Learning and Metric Learning
- The intuition: pull positives together, push negatives apart
- **InfoNCE / NT-Xent loss** (used in SimCSE, DPR):

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q_i, d_i^+)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(q_i, d_j)/\tau)}$$

  where $\tau$ is a temperature hyperparameter
- In-batch negatives: why they're efficient and what their limitations are
- Margin-based losses: triplet loss $\mathcal{L} = \max(0, \|q - d^+\|^2 - \|q - d^-\|^2 + \alpha)$

### 2.4 BERT as an Encoder
- Transformer architecture recap: attention, positional encoding, [CLS] token
- Why BERT representations need fine-tuning for retrieval (anisotropy problem)
- Bi-encoder vs. cross-encoder architectures and their trade-offs

---

## Module 3: Fine-Tuning BERT for Retrieval

### 3.1 Dataset Preparation
- Choosing a dataset: MS MARCO passages, Natural Questions, or a custom domain dataset
- Format: (query, positive passage, negative passage) triplets
- Tokenization with `transformers.AutoTokenizer`
- Building a custom `torch.utils.data.Dataset`

### 3.2 Model Architecture
- Loading a pre-trained BERT: `AutoModel.from_pretrained("bert-base-uncased")`
- Pooling strategies: [CLS] token, mean pooling, max pooling
- Optional projection head: `nn.Linear(hidden_size, embedding_dim)`
- Bi-encoder setup: shared or separate encoders for query and document

### 3.3 Training
- Implementing InfoNCE loss in PyTorch
- In-batch negative mining
- Mixed precision training with `torch.cuda.amp`
- Gradient accumulation for large effective batch sizes
- Logging with TensorBoard or Weights & Biases

### 3.4 Evaluation
- Recall@K and Mean Reciprocal Rank (MRR):

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

- Building a FAISS index for approximate nearest neighbor search
- Running end-to-end retrieval evaluation on a held-out set

---

## Module 4: Hands-On Notebook

A single self-contained Jupyter notebook (`finetune_bert_retrieval.ipynb`) covering:

1. Environment setup (packages, GPU check)
2. Data loading and tokenization
3. Bi-encoder model definition
4. InfoNCE loss implementation
5. Training loop with evaluation callbacks
6. Indexing embeddings with FAISS
7. Interactive query demo

---

## Research Questions to Investigate

1. What pooling strategy yields the best retrieval performance for BERT-base?
2. How sensitive is retrieval quality to the temperature parameter $\tau$ in InfoNCE loss?
3. What is the impact of hard negative mining vs. random in-batch negatives?
4. How does the fine-tuned model compare to a BM25 baseline on out-of-domain queries?

---

## Key References

- Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers*
- Karpukhin et al. (2020) — *Dense Passage Retrieval for Open-Domain QA (DPR)*
- Gao et al. (2021) — *SimCSE: Simple Contrastive Learning of Sentence Embeddings*
- Johnson et al. (2021) — *Billion-scale similarity search with FAISS*
- PyTorch official documentation: https://pytorch.org/docs/stable/index.html
- Hugging Face Transformers: https://huggingface.co/docs/transformers