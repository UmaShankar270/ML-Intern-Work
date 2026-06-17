# UMAP: Uniform Manifold Approximation and Projection

> **Making sense of complex, high-dimensional data — one neighborhood at a time.**

This README takes you from "I've never heard of UMAP" to "I can use and explain UMAP confidently." You'll learn what UMAP is, why it exists, how it works under the hood, when to reach for it, and how to talk about it in an ML interview — no prior machine learning background required.

---

## What Is UMAP?

### Starting From Zero: What Is "Dimensionality"?

Imagine describing a person using height, weight, age, shoe size, and hair color. Each trait is a **dimension** — one measurable property. Stack enough traits together and you get a **data point** living in what mathematicians call a **high-dimensional space**.

Humans think comfortably in 2D (a flat map) or 3D (the physical world). Real datasets rarely stay that polite. A single grayscale photo that's 28×28 pixels — the kind used in handwritten-digit datasets — is technically one point in **784-dimensional space**, one dimension per pixel. Beyond three dimensions, our intuition simply has nothing to grab onto.

**Dimensionality reduction** is the practice of shrinking a dataset from hundreds or thousands of dimensions down to 2 or 3, while keeping as much of the meaningful structure as possible. It's the same trade-off cartographers make when flattening a globe into a paper map: some accuracy is sacrificed, but you gain the ability to see the whole thing at a glance.

### So What Is UMAP, Exactly?

**UMAP (Uniform Manifold Approximation and Projection)** is a dimensionality reduction algorithm published in 2018 by Leland McInnes, John Healy, and James Melville. Its job is to take high-dimensional data and project it into 2D or 3D so that **things that were similar stay close together, and things that were different stay far apart**.

Picture a snow globe full of suspended glitter, each flake representing a data point. Shake it and the glitter floats in three dimensions, clustered in loose clumps. UMAP is the act of pressing that snow globe flat between two glass plates — the clumps get squashed into a 2D pattern, but a clump that was tight in 3D is still recognizably a clump on the flattened glass.

### Why Was It Invented?

Before UMAP, two tools dominated this problem, and each came with a tax:

- **PCA (Principal Component Analysis)** is fast and easy to interpret, but it only sees *straight-line* relationships. If your data bends or folds, PCA flattens it the way a heavy book flattens a crumpled letter — by force, ignoring the creases.
- **t-SNE** captures bends and folds far better than PCA, but it slows to a crawl on large datasets, struggles past roughly a hundred thousand points, and gives you a slightly different picture nearly every time you run it.

UMAP was engineered to close both gaps at once: it runs faster than t-SNE, scales comfortably into the millions of data points, preserves more of the data's overall shape (not just tight local clusters), and produces results that are far more consistent run to run.
| **Visualization** | Exploring clusters, embeddings, and hidden patterns in high-dimensional data | When exact coordinate interpretation is required |

### Real-World Examples

- **Biology / genomics:** A single sequenced cell can carry around 20,000 gene-expression measurements — 20,000 dimensions. UMAP compresses that down to 2D so biologists can literally see which cell types group together, a workflow known as scRNA-seq visualization.
- **Natural language processing:** Word-embedding models like Word2Vec represent each word as a 300-dimensional vector. Run UMAP on those vectors and "king," "queen," "prince," and "princess" land in the same visual neighborhood, even though no one told the algorithm what royalty means.
- **Fraud detection:** Credit card transactions carry dozens of numeric features. Projected with UMAP, fraudulent transactions often form a visibly separate island, distinct from the dense mass of legitimate purchases.
- **Image recognition:** After a neural network learns to classify images, UMAP can plot its internal feature space and confirm that cat images and dog images occupy distinct regions — a sanity check that the network learned something meaningful.
- **Drug discovery:** Molecules are often encoded as long binary fingerprints — 2,048 dimensions or more, each flagging a structural feature. UMAP projects these fingerprints to reveal which drug candidates share chemical "shape," even before any lab testing.

### The Big Analogy

Picture a large sheet of paper that's been crumpled into a ball. The sheet still has neighborhoods — points near each other on the original flat paper, even after crumpling, tend to stay physically close in the crumpled ball. UMAP's task is to carefully **uncrumple and flatten** that ball while preserving those neighborhoods as faithfully as it can. The flattening is never perfect — some folds get stretched, some neighbors drift slightly apart — but UMAP is remarkably good at keeping the paper's local geography intact.

---

## Mathematical Formulation

> Each equation below is a tool, not a hurdle. Every symbol is explained, and no derivations are shown — only the final, usable forms.

### Equation 1: Fuzzy Membership Strength (High-Dimensional Space)

$$w_{ij} = \exp\!\left(\frac{-\max(0,\, d(x_i, x_j) - \rho_i)}{\sigma_i}\right)$$

**Every symbol explained:**

| Symbol | Meaning |
|--------|---------|
| $w_{ij}$ | The connection strength between point $i$ and point $j$ in the original high-dimensional data, ranging from 0 (unrelated) to 1 (essentially the same neighborhood). |
| $\exp(\cdot)$ | The exponential function. It keeps the result positive and makes the strength fall off quickly as distance grows. |
| $d(x_i, x_j)$ | The raw distance between point $i$ and point $j$ in the original space (usually Euclidean). |
| $\rho_i$ | The distance from point $i$ to its single nearest neighbor — a personal "starting line" for that point. It guarantees every point has at least one strong connection, even in a sparse, lonely corner of the data. |
| $\sigma_i$ | A local bandwidth: how far point $i$'s neighborhood is allowed to stretch. It's tuned automatically so every point ends up with roughly the same *effective* number of neighbors, set by the `n_neighbors` hyperparameter. |

**What this means in plain English:** for every point, UMAP looks at its surroundings and converts "how close is this neighbor" into a weight. Very close neighbors score near 1; distant ones score near 0. The $\rho_i$ term is the safety net — it makes sure no point is treated as a total outsider just because it happens to live in a sparser region of the data.

**Why this matters practically:** this is the step where UMAP captures *local* structure. Each point builds its own personalized neighborhood instead of applying one fixed rule to the whole dataset, which is exactly what lets UMAP handle data that's dense in some regions and sparse in others without distorting either.

---

### Equation 2: Low-Dimensional Similarity

$$v_{ij} = \left(1 + a \cdot \|y_i - y_j\|^{2b}\right)^{-1}$$

**Every symbol explained:**

| Symbol | Meaning |
|--------|---------|
| $v_{ij}$ | The connection strength between the *projected* (2D or 3D) positions of points $i$ and $j$. |
| $y_i, y_j$ | The coordinates of points $i$ and $j$ in the low-dimensional output — the plot you actually see. |
| $\|\cdot\|$ | Euclidean distance between those two projected positions. |
| $a, b$ | Shape parameters, fitted automatically, that control how sharply similarity drops off as projected distance increases. By default they approximate a Student-t curve. |

**What this means in plain English:** once points have a tentative position on the 2D plot, this formula converts their plotted distance back into a similarity score — close points score near 1, far points score near 0. It is the low-dimensional mirror of Equation 1.

**Why this matters practically:** this particular curve shape lets points that genuinely belong far apart actually spread out in the plot, rather than getting crushed into one overcrowded blob. That property is what reduces the "crowding problem" that made earlier methods like t-SNE struggle to show global structure.

---

### Equation 3: Cross-Entropy Loss (The Objective)

$$\mathcal{L} = \sum_{(i,j) \in E} \left[ w_{ij} \log\!\frac{w_{ij}}{v_{ij}} + (1 - w_{ij}) \log\!\frac{1 - w_{ij}}{1 - v_{ij}} \right]$$

**Every symbol explained:**

| Symbol | Meaning |
|--------|---------|
| $\mathcal{L}$ | The total loss — a single number measuring how badly the 2D layout disagrees with the original high-dimensional neighborhood structure. UMAP's entire job is to make this number small. |
| $\sum_{(i,j) \in E}$ | A sum over every pair of points connected by an edge in the neighborhood graph $E$. |
| $w_{ij}$ | High-dimensional connection strength, from Equation 1. |
| $v_{ij}$ | Low-dimensional connection strength, from Equation 2. |
| $\log$ | The natural logarithm, a standard way to compare two probability-like values. |

**What this means in plain English:** this is UMAP's report card. A loss of zero would mean the 2D layout perfectly mirrors the original neighborhood structure — an unreachable ideal in practice. UMAP uses gradient descent (small, repeated nudges to the point positions) to push this number as low as it can.

**Why this matters practically:** cross-entropy is one of the most battle-tested loss functions in machine learning, also used to train classification networks. Borrowing it here means UMAP's optimization step inherits decades of well-understood, efficient tooling rather than inventing a bespoke solver from scratch.

---

## How It Works (Step-by-Step)
![UMAP Workflow](images/umap_workflow.png)

*Figure 1: High-dimensional data → Neighbor graph → Fuzzy simplicial set → Optimization → Low-dimensional embedding.*

### Step 1: Compute the Neighborhood Graph

**What happens:** for every data point, UMAP finds its `n_neighbors` closest points in the original high-dimensional space, using a fast approximate nearest-neighbor search.

**Why:** comparing every point to every other point would be computationally ruinous at scale. Approximate search (commonly NN-Descent) finds "close enough" neighbors in a fraction of the time an exhaustive search would take.

**Intuition:** imagine a school reunion where every attendee is asked to name their 15 closest old friends in the room. Nobody is mapping the entire social web yet — just gathering local friendships, one person at a time.

**Example:** across 50,000 handwritten-digit images (784 dimensions each), this step finds, for every image, the 15 most visually similar images — other 3s with the same slant, other 7s with the same stroke width.

---

### Step 2: Assign Fuzzy Edge Weights

**What happens:** using Equation 1, UMAP scores every neighborhood connection between 0 and 1, then symmetrizes the scores with $W_{ij} = w_{ij} + w_{ji} - w_{ij} \cdot w_{ji}$.

**Why:** not all neighbors are equally close — a score of 0.9 means "nearly identical," a score of 0.1 means "technically neighbors, barely related." Symmetrizing ensures the connection between $i$ and $j$ doesn't depend on which point you ask first.

**Intuition:** think of two coworkers rating how close their working relationship is. If Priya rates her bond with Arjun as 0.9 but Arjun rates it as 0.6, the formula above blends both perspectives into one final number — neither viewpoint gets silently discarded.

---

### Step 3: Initialize the Low-Dimensional Layout

**What happens:** UMAP places every point in 2D using spectral embedding — a mathematically guided starting layout, far better than scattering points at random.

**Why:** a sensible starting point dramatically speeds up convergence and makes the final result more stable across runs. Random initialization forces the optimizer to do far more work for a worse result.

**Intuition:** a city planner doesn't drop buildings onto a blank map at random and rearrange from there. They sketch zones first — residential here, industrial there — based on known relationships, then refine. UMAP's spectral initialization is that first rough sketch.

---

### Step 4: Optimize with Stochastic Gradient Descent

**What happens:** UMAP repeatedly nudges every point's 2D position to shrink the cross-entropy loss from Equation 3. Connected points are pulled together; unconnected points are pushed apart. To keep this fast, it uses **negative sampling** — randomly sampling a small set of non-neighbor pairs for the repulsion step instead of checking every possible pair.

**Why:** updating every point based on every other point would cost O(n²) work per pass — unworkable at scale. Negative sampling approximates the repulsion using a small random sample, which is what keeps UMAP's runtime close to linear.

**Intuition:** picture a tug-of-war between rubber bands and stretched springs. Rubber bands link points that belong together, constantly pulling them inward. Springs push unrelated points apart whenever they drift too close. The layout settles once the pulling and pushing roughly balance out — that settled state is your final UMAP plot.

---

### Step 5: Return the Projection

**What happens:** after a fixed number of epochs (full optimization passes), UMAP outputs the final 2D or 3D coordinates for every point.

**Why:** these coordinates are what gets plotted, fed into downstream models, or stored as compressed features.

**Example:** the original 50,000 images × 784 pixels becomes 50,000 rows × 2 columns. Each row is now a single (x, y) point on a scatter plot, and images of the same digit cluster together visually.

---

## Key Assumptions

### Assumption 1: The Data Lies on a Manifold

**What it means:** UMAP assumes the high-dimensional data doesn't fill space randomly — it lives on a lower-dimensional surface, called a **manifold**, embedded inside that larger space. A crumpled sheet of paper is a 2D manifold sitting inside 3D space; the surface of a globe is the same idea.

**Why UMAP relies on it:** the algorithm's entire mathematical foundation — Riemannian geometry, the study of curved surfaces — depends on this assumption. Without it, the neighborhood graph and the distances UMAP computes lose their theoretical grounding.

**What happens if violated:** if the data genuinely is random noise with no underlying surface, UMAP will still draw a plot — it just won't mean anything. Apparent clusters may be artifacts of the algorithm, not real structure in the data.

---

### Assumption 2: The Manifold Is Locally Connected

**What it means:** for each point, UMAP assumes its immediate neighborhood is roughly flat — the same way a small patch of Earth's surface looks flat to someone standing on it, even though the planet as a whole is a sphere.

**Why UMAP relies on it:** this lets UMAP measure local distances using ordinary Euclidean geometry, even though the data's overall shape may be curved or twisted.

**What happens if violated:** if the data has sharp gaps or disconnected islands with no smooth path between them, the *relative positions* of distant clusters in the final plot may not be trustworthy — though structure within each cluster usually still is.

---

### Assumption 3: `n_neighbors` Reflects True Local Scale

**What it means:** UMAP assumes the `n_neighbors` value you choose is a reasonable description of each point's actual neighborhood — not so small that it's just noise, not so large that it blurs distinct groups together.

**Why UMAP relies on it:** the bandwidth $\sigma_i$ from Equation 1 is tuned specifically to make each point's effective connectivity match `n_neighbors`. Pick a value that's badly mismatched to your data's real density, and the resulting graph misrepresents the actual structure.

**What happens if violated:** too small, and UMAP overreacts to noise, fragmenting the data into spurious micro-clusters. Too large, and it smooths over real boundaries, merging groups that should stay separate. Most datasets behave reasonably somewhere in the 5–50 range.

---

## When to Use / When Not to Use

| Criterion | ✅ Use UMAP | ❌ Avoid UMAP |
|-----------|------------|--------------|
| **Goal** | Visualizing high-dimensional data in 2D or 3D | You need mathematically exact distance preservation |
| **Dataset size** | Large datasets (10K–10M+ rows) | Very small datasets (under ~500 rows), where simpler methods are just as good |
| **Data structure** | Complex, nonlinear, manifold-like data | Data that's already linearly separable — PCA is faster and just as clear |
| **Downstream task** | Generating embeddings for clustering, classification, or search | You need fully interpretable axes — UMAP's axes carry no inherent meaning |
| **Speed needs** | You need faster results than t-SNE on large data | You need a deterministic, identical-every-time result without fixing a random seed |
| **Cluster exploration** | Exploring unknown structure in a new dataset | Rigorous statistical inference — UMAP is an exploratory tool, not a hypothesis test |
| **Transfer / transform** | You need to project new points into an existing embedding | Straightforward feature engineering, where PCA or autoencoders are more established |
| **Examples** | Single-cell RNA sequencing, NLP embeddings, image feature spaces, anomaly detection | Linear regression preprocessing, small tabular datasets, formal statistical testing |

---

## Implementation Overview

### Conceptual Workflow

Regardless of whether you build it yourself or use a library, the underlying logic is the same:

```
Raw Data (N × D)
    → Build nearest-neighbor graph
    → Compute fuzzy edge weights
    → Initialize 2D layout (spectral embedding)
    → Optimize layout (stochastic gradient descent)
    → Output (N × 2)
```

---

### From Scratch (NumPy / Pure Python)

| Aspect | Details |
|--------|---------|
| **What you build** | A KNN graph via brute-force or ball-tree search, edge weights via Equation 1, a spectral initialization via eigendecomposition, and an SGD loop applying attractive and repulsive forces |
| **Advantages** | Deep, first-hand understanding of every moving part; fully customizable; zero library dependencies; excellent for research and learning |
| **Limitations** | Brute-force KNN is O(n² × d) — painfully slow; no approximate nearest-neighbor speedups; no GPU support; realistically weeks of engineering to get production-grade |
| **Production suitability** | Not suitable for production. Strictly educational. |
| **When to attempt** | Writing a thesis, contributing to dimensionality-reduction research, or wanting to genuinely understand the algorithm's internals rather than just its API |

---

### Using `umap-learn` (Recommended)

| Aspect | Details |
|--------|---------|
| **What you get** | A scikit-learn-style API (`.fit()`, `.transform()`, `.fit_transform()`) backed by NN-Descent for fast approximate KNN and Numba-JIT-compiled optimization, with optional GPU acceleration |
| **Advantages** | Handles millions of points; reproducible results when `random_state` is fixed; supports supervised UMAP (using labels to guide the projection) and Parametric UMAP (a neural-network-based variant built for generalization) |
| **Limitations** | Internals are a black box for most users; results are approximate, not exact; sensitive to hyperparameter choices; output axes carry no interpretable meaning |
| **Production suitability** | Fully production-ready, and used in real industry pipelines — for instance in genomics platforms for single-cell analysis and in audio/recommendation systems for embedding visualization |
| **Key hyperparameters** | `n_neighbors` (local vs. global balance), `min_dist` (cluster tightness), `n_components` (output dimensions), `metric` (distance function) |

---

### Hyperparameter Intuition

- **`n_neighbors` (default: 15):** controls the local-vs-global trade-off. Small values produce tight micro-clusters but lose sight of the bigger picture. Large values reveal more global shape but risk merging clusters that should stay distinct.
- **`min_dist` (default: 0.1):** controls how tightly points pack together in the 2D plot. Small values give dense, compact clusters; larger values spread points out more evenly, making the overall topology easier to see.
- **`metric`:** use `'euclidean'` for general numerical data, `'cosine'` for text embeddings, and `'hamming'` for binary data like molecular fingerprints.

---

## Top 5 Interview Questions

### Q1: "How is UMAP different from t-SNE? When would you choose one over the other?"

**Strong Answer:**

Both are nonlinear dimensionality reduction methods built to preserve local neighborhood structure, but they diverge on speed, scalability, and global structure.

t-SNE, even with Barnes-Hut optimization, starts struggling past roughly 100K points. UMAP behaves close to linearly in practice and routinely handles millions of points. More importantly, UMAP's underlying framework — built on fuzzy topology — explicitly tries to preserve both local *and* global structure, while t-SNE is almost entirely local; the distances between clusters on a t-SNE plot are not meaningful.

I'd reach for t-SNE on smaller datasets, in domains where it's been validated extensively (some biology communities still favor it for certain flow-cytometry data), or when run-to-run reproducibility matters less. I'd reach for UMAP on large datasets, when the relative positions of clusters matter, when I need to project new points into an existing embedding (UMAP supports this natively; vanilla t-SNE does not), or when the embedding feeds into a downstream model.

**Key points interviewer expects:** speed comparison, scalability, global vs. local structure, transform capability, Parametric UMAP.

---

### Q2: "What does `n_neighbors` control, and what are the trade-offs?"

**Strong Answer:**

`n_neighbors` sets how many neighboring points each data point considers when building its local connectivity graph, and it governs the core local-versus-global trade-off in the algorithm.

At low values (say, 5), each point only "sees" a tiny neighborhood, so UMAP becomes hypersensitive to local microstructure and can fragment the data into many small clusters — some real, some just noise. At high values (say, 100), each point considers a broad neighborhood, giving a better sense of global topology but risking the merger of clusters that are genuinely distinct in the original space.

In practice I start at the default of 15 and tune from there: lower if I suspect fine-grained subpopulations exist, higher if the data shows diffuse, overlapping clusters. I also watch for thin "hair" or "bridge" artifacts connecting clusters in the plot — that's usually a sign `n_neighbors` is set too low.

**Key points interviewer expects:** local vs. global trade-off, practical tuning strategy, connection to the $\sigma_i$ bandwidth parameter.

---

### Q3: "Can UMAP be used as a preprocessing step for a classifier? What are the risks?"

**Strong Answer:**

Yes, and it's a common pattern, especially in genomics and NLP pipelines — compress high-dimensional features down to roughly 10–50 dimensions with UMAP, then feed that into a classifier like XGBoost or a neural network.

The upside: substantial dimensionality reduction, some noise removal, and occasionally better cluster separability for the downstream model. The risks are real, though. UMAP is a lossy, nonlinear transform, so information irrelevant to the projection might still matter for classification. Standard UMAP also doesn't generalize perfectly to brand-new data — the `.transform()` step is itself an approximation. And supervised UMAP, which uses labels during the projection, can quietly inflate evaluation accuracy if the labels leak into the embedding before a proper train/test split.

For production classification pipelines, Parametric UMAP is the safer choice — it learns a neural network to perform the projection, which generalizes to new samples properly. For purely exploratory work, standard UMAP as a preprocessing step is fine.

**Key points interviewer expects:** lossy compression caveat, generalization issues, Parametric UMAP, supervised vs. unsupervised modes, data leakage risks.

---

### Q4: "What is UMAP's mathematical foundation, and how does it differ from t-SNE's?"

**Strong Answer:**

t-SNE's foundation is purely probabilistic. It converts distances into conditional probabilities — Gaussian in the high-dimensional space, Student-t in the low-dimensional space — and minimizes the KL divergence between the two distributions.

UMAP's foundation is more geometric, rooted in Riemannian geometry and algebraic topology, specifically the theory of fuzzy simplicial sets. The working assumption is that the data is sampled from a manifold with an unknown local metric; UMAP estimates that metric around each point and builds a fuzzy topological representation of the whole dataset. The 2D projection is then found by minimizing cross-entropy between the high- and low-dimensional fuzzy representations, rather than KL divergence.

That theoretical difference has a real practical consequence: UMAP's fuzzy-set framework tends to balance local and global structure more evenly, while t-SNE's KL divergence punishes "crowding" far more harshly than separation — which is part of why t-SNE plots often show tight, isolated clusters with exaggerated empty space between them.

**Key points interviewer expects:** Riemannian geometry, fuzzy simplicial sets, cross-entropy vs. KL divergence, practical implications of the theoretical difference.

---

### Q5: "How would you validate a UMAP embedding in a production ML pipeline?"

**Strong Answer:**

Validation is genuinely tricky here since there's no single ground truth, so I combine several checks.

First, visual sanity checks: do known similar items land near each other? In a text-embedding use case, do documents on the same topic cluster together?

Second, trustworthiness and continuity scores, which quantify how well local neighborhoods survive the projection. Trustworthiness asks whether a point's 2D neighbors were also its neighbors in the original space; continuity asks the reverse. Both range from 0 to 1, and values above roughly 0.9 are generally considered solid.

Third, downstream task performance — if the embedding feeds a classifier or clustering step, I compare that task's results with and without UMAP preprocessing.

Fourth, stability across random seeds — running UMAP 5–10 times and checking whether the qualitative shape (cluster boundaries, relative positions) stays consistent. High variance across runs is a red flag for over-sensitivity to initialization.

Finally, in production, I always version the fitted UMAP transformer itself, so new data gets projected consistently into the same embedding space rather than drifting over time.

**Key points interviewer expects:** trustworthiness/continuity metrics, downstream validation, stability testing, model versioning, train/test separation.

---

## Quick Reference Table

| Item | Details |
|------|---------|
| **Algorithm Type** | Nonlinear dimensionality reduction / manifold learning |
| **Learning Type** | Unsupervised by default; supervised mode available when labels are provided |
| **Time Complexity** | Approximately O(n^1.14) empirically — close to linear, thanks to approximate KNN (NN-Descent) and negative sampling |
| **Space Complexity** | O(n × n_neighbors) for storing the neighborhood graph |
| **Main Hyperparameters** | `n_neighbors` (5–50), `min_dist` (0.0–1.0), `n_components` (typically 2 or 3), `metric` (`'euclidean'`, `'cosine'`, `'manhattan'`, etc.), `n_epochs` (200–500) |
| **Output** | An N × n_components matrix of projected coordinates |
| **Initialization** | Spectral embedding by default, or random |
| **Strengths** | Fast at scale, preserves both local and global structure, supports transforming new data, flexible distance metrics, actively maintained |
| **Weaknesses** | Axes carry no inherent meaning, results vary run-to-run without a fixed seed, sensitive to hyperparameters, theoretical guarantees are approximate |
| **Compared to PCA** | Captures nonlinear structure that PCA misses; PCA remains faster and more interpretable for linear data |
| **Compared to t-SNE** | Faster, better at preserving global structure, supports transforming new data; t-SNE can still produce cleaner local clusters on smaller datasets |
| **Production Library** | `umap-learn` (`pip install umap-learn`) |
| **GPU Support** | Yes, via `cuML` (the RAPIDS library) |
| **Parametric Variant** | Parametric UMAP — a neural-network encoder that generalizes fully to new data |

---

## References & Further Reading

### Original Paper
- **McInnes, L., Healy, J., & Melville, J. (2018).** *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* arXiv:1802.03426. Dense but authoritative — the introduction and early sections are approachable even for newcomers.
  - 🔗 https://arxiv.org/abs/1802.03426

### Official Documentation
- **umap-learn documentation.** Covers installation, every hyperparameter, supervised UMAP, Parametric UMAP, and practical how-to guides, written by the algorithm's own authors.
  - 🔗 https://umap-learn.readthedocs.io/en/latest/

### High-Quality Tutorial
- **"How UMAP Works" — official conceptual overview**, by Leland McInnes. One of the most approachable explanations of the underlying math, aimed at practitioners rather than theoreticians.
  - 🔗 https://umap-learn.readthedocs.io/en/latest/how_umap_works.html

### Kaggle Resource
- **"UMAP Dimension Reduction — Let's Practice."** A hands-on Kaggle notebook demonstrating UMAP on real datasets, including MNIST, with visualizations and parameter-tuning examples.
  - 🔗 https://www.kaggle.com/code/jpmiller/umap-dimension-reduction-let-s-practice

### Additional Learning
- **"Dimensionality Reduction for Data Visualization" — Towards Data Science.** A side-by-side comparison of PCA, t-SNE, and UMAP on shared datasets, useful for building intuition about when each method wins.
  - 🔗 https://towardsdatascience.com/dimensionality-reduction-for-data-science-interviews-7e3d82be392d

### Common Datasets Used with UMAP

- MNIST
- Fashion-MNIST
- CIFAR Embeddings
- Word2Vec Embeddings
- Single Cell RNA Sequencing Data