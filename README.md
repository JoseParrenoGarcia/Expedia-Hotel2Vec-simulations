# Expedia Hotel2Vec Paper — Simulation Companion

This repository contains **toy simulations** that accompany an analysis of Expedia Group’s paper:

> **“Hotel2vec: Learning Attribute-Aware Hotel Embeddings with Self-Supervision”**  
> Ashkan Sadeghian, Aishwarya Agrawal, Brian D. Davison — 2019  
> https://arxiv.org/abs/1910.03943

The goal of this repo is **not** to reproduce Expedia’s production recommendation system, nor to provide a reusable embedding library.  
Instead, it exists to **make the mechanics of the Hotel2Vec paper visible, intuitive, and inspectable**.

---

## What this repository is

### 1. A companion to technical analysis and learning

All simulations in this repository are written to support **conceptual understanding** of the Hotel2Vec paper.

They are intended to help answer questions such as:
- What signal does session-based skip-gram actually capture?
- How does side information reshape the embedding space?
- Why does attribute-aware training help with cold start?
- What changes geometrically when content and behaviour are combined?

Each script corresponds to a **specific modelling choice or idea** introduced in the paper.

---

### 2. A practical breakdown of the Hotel2Vec paper

The Hotel2Vec paper extends classic word2vec-style objectives to the travel domain, introducing:

- Session-based hotel embeddings learned from click sequences
- Skip-gram objectives adapted to sparse, noisy travel behaviour
- Attribute-aware embeddings using structured hotel metadata
- Cold-start handling via side-information alignment
- Joint representation of behavioural and content signals

The simulations here are **directly inspired by the paper’s modelling ideas**, but are deliberately simplified so that:

- Individual forces in the loss are easy to reason about
- Geometry can be inspected directly
- Behaviour emerges in small, understandable examples

---

### 3. Toy simulations, not production code

All simulations in this repository are **toy examples by design**.

They typically:
- Use very small datasets or synthetic sessions
- Run on a single machine with minimal optimisation
- Focus on clarity over performance or scalability
- Prefer explicit loops and simple objectives to abstractions

This makes them suitable for:
- Building intuition
- Teaching or self-study
- Rapid experimentation with modelling assumptions

It also means they are **not suitable for training real embeddings**.

---

## Repository structure

Below is a high-level overview of the main simulations and what they illustrate.

---

### Click-only skip-gram baseline
**`skip_gram_model_only_clicks.py`**

Simulates the core Hotel2Vec idea: learning hotel embeddings purely from **click sessions**, using a skip-gram-style objective.

Key idea:
- Hotels clicked within the same session are pulled together
- The embedding space reflects behavioural co-occurrence
- No side information is used

This serves as the **baseline behavioural model**.

---

### Attribute-aware skip-gram model
**`skip_gram_model_full.py`**

Extends the click-only model by incorporating **hotel attributes** (side information) into the embedding process.

Key idea:
- Structured attributes act as an additional learning signal
- Behavioural and content information jointly shape the space
- Helps regularise embeddings and reduce sparsity issues

This mirrors the paper’s core contribution: **attribute-aware self-supervision**.

---

### Data preparation and session handling
**`data.py`**

Utilities for:
- Loading or generating session data
- Encoding hotel attributes
- Preparing inputs for skip-gram training

Kept intentionally simple to keep focus on modelling rather than pipelines.

---

### Diagnostics and inspection
**`diagnostics.py`**

Helper functions to inspect:
- Training behaviour
- Embedding similarity structure
- Sanity checks on learned representations

Designed to support *thinking*, not benchmarking.

---

### Visualisation utilities
**`plots.py`**

Plotting helpers to:
- Visualise embedding spaces
- Inspect neighbourhood structure
- Observe qualitative differences between models

Visualisation is treated as a **first-class tool for understanding**.

---

## How to use this repository

This repository is best used alongside the paper.

Suggested workflow:
1. Read a section of the Hotel2Vec paper
2. Identify the modelling idea being introduced
3. Open the corresponding simulation script
4. Run it and inspect the outputs
5. Modify assumptions and observe how the embedding geometry changes

The scripts are intentionally small and readable, and are meant to be edited.

---

## What this repository is not

- ❌ A reproduction of Expedia’s production system  
- ❌ A scalable embedding training framework  
- ❌ A recommendation library  

It is a **thinking and learning tool**, not a deployment artefact.

---

## References

Primary reference:
- **Hotel2vec: Learning Attribute-Aware Hotel Embeddings with Self-Supervision**  
  https://arxiv.org/abs/1910.03943

---

## License and attribution

All simulations are original implementations inspired by the public Hotel2Vec paper.

Please cite the original paper if you reuse ideas in academic or professional work.
