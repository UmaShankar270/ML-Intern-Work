# Module 07: Principal Component Analysis (PCA) for Dimensionality Reduction

**Tagline:** Compress the noise, keep the signal — the definitive guide to PCA for learners and practitioners.

**Learning Summary:** What PCA is, why dimensionality reduction is critical, how to calculate covariance and eigenvalues, how to project features onto principal components, and how to implement this in Python from scratch and using Scikit-learn.

---

## Lesson 7.1: The Intuition of Dimensionality Reduction

### What is Dimensionality Reduction?
Dimensionality reduction is the process of compressing a dataset containing many columns (dimensions) into a smaller, more manageable set of columns while preserving as much of the underlying information as possible.

### Why is it Needed?
Modern datasets are often high-dimensional. For example, genomics datasets track thousands of gene expression levels, and image processing tasks represent each pixel as a feature. This abundance of features leads to several critical issues:
1.  **Overfitting:** High-dimensional models have too many degrees of freedom and easily memorize training noise.
2.  **Computational Bottlenecks:** Training time and memory footprints scale exponentially with feature count.
3.  **Visualization Limits:** Humans can only visualize data in 2D or 3D. Higher dimensions are impossible to plot directly.
4.  **Curse of Dimensionality:** As dimensions increase, data points become extremely sparse in the high-dimensional space, rendering distance metrics (like Euclidean distance) less meaningful.

### PCA's Core Concept
Principal Component Analysis (PCA) is an **unsupervised linear feature extraction** technique. It identifies the directions of maximum variance in the high-dimensional space and projects the data onto these directions, creating a set of new, uncorrelated features called **principal components**.

```
[Diagram: High-Dimensional Data Cloud]
         |      .   .   .
         |    .   .  .
         |  .   .  .
         +-------------------
         
[Diagram: PCA Projection Axis Selection]
         |      .   .   .  <-- Project points onto this best-fit axis
         |    /   .  .
         |  /   .  .
         +-------------------
```

### Real-World Analogies

#### Analogy 1: Textbook Summarization
Imagine you have a 500-page chemistry textbook. You need to write a 10-page study guide. You do not randomly select 10 pages and throw away the rest (that would be **Feature Selection**). Instead, you read the textbook and write summary paragraphs that capture the core concepts (this is **Feature Extraction**). You lose the minor details, but you preserve the main signal in a compressed format.

#### Analogy 2: 3D Sculpture Photography
Imagine a complex 3D sculpture in a museum. You want to capture its shape on a flat 2D photograph. If you take a picture from a random angle, the shadows might overlap and obscure the details. PCA is the process of finding the mathematically "optimal camera angle" that displays the greatest spread (variance) of the sculpture, capturing its structure in 2D with minimal overlap.

---

## Lesson 7.2: Mathematical Blueprints & Equations

Before implementing PCA, we must define the mathematical operations that convert raw features into principal components.

### Variables & Symbols Mapping

| Symbol | Representation | Dimensionality | Conceptual Meaning |
| :--- | :--- | :--- | :--- |
| $X$ | Raw Data Matrix | $n \times d$ | Input dataset with $n$ rows (samples) and $d$ columns (original features). |
| $\mu$ | Feature Mean Vector | $1 \times d$ | Row vector containing the mean of each of the $d$ features. |
| $\tilde{X}$ | Centered Data Matrix | $n \times d$ | The scaled dataset shifted to be centered around the origin. |
| $\Sigma$ | Covariance Matrix | $d \times d$ | Symmetric matrix representing how all features vary in relation to each other. |
| $\mathbf{v}_i$ | Eigenvector | $d \times 1$ | A direction vector in feature space defining a principal component. |
| $\lambda_i$ | Eigenvalue | Scalar | The variance (information) captured along the corresponding eigenvector $\mathbf{v}_i$. |
| $W$ | Projection Matrix | $d \times k$ | Matrix containing the top $k$ sorted eigenvectors as columns. |
| $Z$ | Projected Data Matrix | $n \times k$ | The compressed dataset representing the original data in $k$ dimensions. |

---

### Core Equations

#### 1. Mean Centering
$$\tilde{X} = X - \mu$$
Subtracting the mean vector shifts the data cloud so that its center of mass lies exactly at the origin $(0, 0, \dots, 0)$.
> [!IMPORTANT]
> PCA measures variance as the distance of data points from the center of the cloud. Skipping mean centering causes the first principal component to point toward the center of the dataset instead of the direction of maximum spread.

#### 2. Covariance Matrix
$$\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X}$$
This computes the pairwise relationship between all features. The diagonal of this $d \times d$ matrix holds the variance of individual features, while the off-diagonal elements hold their covariances.
> [!NOTE]
> Positive covariance means features move together; negative means they move in opposite directions; zero means they are uncorrelated.

#### 3. Eigen Decomposition
$$\Sigma \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
This equation breaks down the covariance matrix into eigenvectors ($\mathbf{v}_i$) and eigenvalues ($\lambda_i$). 
*   The **eigenvector** represents the direction of the new coordinate axes.
*   The **eigenvalue** represents the magnitude of the data's spread (variance) along that axis.

#### 4. Component Projection
$$Z = \tilde{X} W$$
This step transforms our dataset. By multiplying our centered data matrix $\tilde{X}$ by the projection matrix $W$ (which holds the top $k$ eigenvectors), we obtain a compressed matrix $Z$ containing only $k$ columns.

#### 5. Explained Variance Ratio (EVR)
$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$
This ratio calculates the percentage of total dataset variance captured by the $i$-th component. It provides a mathematical basis for choosing how many dimensions to keep.

---

### Worked Numerical Example (2D Dataset)

Let's compute the covariance matrix and principal directions for a simple 2D dataset with $n=3$ samples and $d=2$ features.

#### Step 1: Raw Data Matrix ($X$)
$$X = \begin{bmatrix} 2.0 & 3.0 \\ 4.0 & 5.0 \\ 6.0 & 7.0 \end{bmatrix}$$

#### Step 2: Calculate Feature Means ($\mu$)
*   $\mu_1 = \frac{2 + 4 + 6}{3} = 4.0$
*   $\mu_2 = \frac{3 + 5 + 7}{3} = 5.0$
$$\mu = \begin{bmatrix} 4.0 & 5.0 \end{bmatrix}$$

#### Step 3: Compute Centered Data ($\tilde{X}$)
$$\tilde{X} = X - \mu = \begin{bmatrix} 2 - 4 & 3 - 5 \\ 4 - 4 & 5 - 5 \\ 6 - 4 & 7 - 5 \end{bmatrix} = \begin{bmatrix} -2.0 & -2.0 \\ 0.0 & 0.0 \\ 2.0 & 2.0 \end{bmatrix}$$

#### Step 4: Compute Covariance Matrix ($\Sigma$)
$$\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X} = \frac{1}{2} \begin{bmatrix} -2.0 & 0.0 & 2.0 \\ -2.0 & 0.0 & 2.0 \end{bmatrix} \begin{bmatrix} -2.0 & -2.0 \\ 0.0 & 0.0 \\ 2.0 & 2.0 \end{bmatrix}$$
$$\Sigma = \frac{1}{2} \begin{bmatrix} ((-2)^2 + 0^2 + 2^2) & ((-2)(-2) + 0 + (2)(2)) \\ ((-2)(-2) + 0 + (2)(2)) & ((-2)^2 + 0^2 + 2^2) \end{bmatrix}$$
$$\Sigma = \frac{1}{2} \begin{bmatrix} 8.0 & 8.0 \\ 8.0 & 8.0 \end{bmatrix} = \begin{bmatrix} 4.0 & 4.0 \\ 4.0 & 4.0 \end{bmatrix}$$

#### Step 5: Solve for Eigenvalues and Eigenvectors
Solving $\det(\Sigma - \lambda I) = 0$:
$$\det \begin{bmatrix} 4 - \lambda & 4 \\ 4 & 4 - \lambda \end{bmatrix} = 0$$
$$(4 - \lambda)^2 - 16 = 0 \implies \lambda^2 - 8\lambda = 0$$
*   **Eigenvalues:** $\lambda_1 = 8.0$, $\lambda_2 = 0.0$
*   **Eigenvectors:** For $\lambda_1 = 8.0$, the eigenvector is $\mathbf{v}_1 = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} 0.7071 \\ 0.7071 \end{bmatrix}$

This indicates that $100\%$ of the variance ($\frac{8}{8+0}$) lies along a single diagonal line, allowing us to compress this 2D dataset to 1D with zero information loss.

---

## Lesson 7.3: How PCA Works — Step-by-Step

```
[Raw Data] ──> [Standardization] ──> [Covariance Matrix] ──> [Eigen Decomposition]
                                                                     │
[Projected Data] <── [Matrix Multiplication] <── [Select Top k] <────┘
```

### Step 1: Collect Data
Gather $n$ samples into a matrix $X$ of size $n \times d$.

### Step 2: Standardize Features (Z-Score Normalization)
Ensure every feature has a mean of 0 and standard deviation of 1.
$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$
> [!WARNING]
> Without standardization, features with larger numeric scales (e.g. income in thousands) will dominate the variance calculations over features with smaller scales (e.g. age in years).

### Step 3: Compute Covariance Matrix
Construct the $d \times d$ covariance matrix $\Sigma$ to capture linear correlations between all feature pairs.

### Step 4: Solve for Eigenvalues and Eigenvectors
Use numerical solvers (like SVD or eigen decomposition) to compute the eigenvectors and eigenvalues of $\Sigma$.

### Step 5: Rank Principal Components
Sort the eigenvectors by their eigenvalues in descending order. The eigenvector with the largest eigenvalue is the first principal component (PC1).

### Step 6: Select the Target Dimensions ($k$)
Choose how many components ($k$) to keep. We plot a **Scree Plot** showing the explained variance per component and select $k$ where the "elbow" occurs, or select $k$ such that the cumulative explained variance exceeds a threshold (typically $95\%$).

```
[Scree Plot Placeholder]
Explained
Variance (%)
  ▲
40│  ● (PC1: 36%)
30│
20│       ● (PC2: 19%)
10│            ● (PC3: 11%)
 0└────────────────────────►
     PC1  PC2  PC3  PC4  PC5
```

### Step 7: Construct the Projection Matrix ($W$)
Stack the top $k$ eigenvectors as columns to form the $d \times k$ projection matrix $W$.

### Step 8: Project Raw Data
Multiply the centered data matrix by the projection matrix: $Z = \tilde{X} W$. The output $Z$ is the compressed dataset of shape $n \times k$.

---

## Lesson 7.4: Core Assumptions & Limitations

Like all statistical methods, PCA makes assumptions about the structure of your data.

### 1. Linearity
PCA assumes that features are linearly related. It projects data onto straight lines in the high-dimensional space. If the data lies on a curved surface (such as the concentric circles shown below), PCA will fail to find a clean separation.

```
[Non-linear Concentric Circles]
       .  *  *  .
     *   .    .   *      <-- PCA cannot separate these with a straight line.
     *   .    .   *          Use Kernel PCA or UMAP instead.
       .  *  *  .
```

### 2. Variance Equals Information
PCA assumes that directions with the largest spread contain the most important signal, while directions with low spread are noise. This assumption holds for many datasets, but it can fail if the key pattern you are looking for lies along a direction of low variance.

### 3. Orthogonality
PCA constrains principal components to be perpendicular (orthogonal) to one another. While this eliminates correlation between components, it may force PCA to choose projection directions that do not align with the true physical axes of variation.

### 4. Interpretability Trade-off
Because principal components are linear combinations of all original features, they do not have a simple real-world meaning. If your business or research requires explainable features, PCA may not be the right choice.

---

## Lesson 7.5: Dimensionality Reduction Comparison Matrix

To help choose the right method, compare the characteristics of the three main dimensionality reduction techniques:

| Feature | Principal Component Analysis (PCA) | Linear Discriminant Analysis (LDA) | t-Distributed Stochastic Neighbor Embedding (t-SNE) |
| :--- | :--- | :--- | :--- |
| **Supervised?** | No (Unsupervised) | Yes (Supervised) | No (Unsupervised) |
| **Linearity** | Linear | Linear | Non-linear |
| **Primary Goal** | Maximize variance | Maximize class separability | Preserve local neighborhoods |
| **Label Required?**| No | Yes | No |
| **Component Limit**| Up to $d$ (original features) | Up to $C - 1$ ($C$ is class count) | Typically 2 or 3 |
| **Determinism** | Deterministic | Deterministic | Stochastic (run-dependent) |
| **Downstream Use** | Modeling, compression | Classification preprocessing | Visualization only |

---

## Lesson 7.6: Placement & Interview Q&A

**Q1: What is PCA and why is it preferred over raw feature pruning?**  
**Answer:** PCA is an unsupervised linear dimensionality reduction method. Instead of discarding original columns (which can lose important information), PCA projects the entire feature space onto orthogonal directions of maximum variance. This compresses the dataset while retaining as much of the original information as possible.

**Q2: What is the purpose of standardizing data before running PCA?**  
**Answer:** PCA is highly sensitive to the scale of features. Features with larger ranges (e.g., income) have higher variance than features with smaller ranges (e.g., age). Without standardization, the first principal component will align almost entirely with the feature that has the largest scale, ignoring the others.

**Q3: How do you choose the optimal number of components ($k$) for a pipeline?**  
**Answer:** There are two main strategies:
1.  **Explained Variance Threshold:** Sum the explained variance ratios of sorted components and select $k$ where the cumulative sum exceeds a threshold (typically $95\%$).
2.  **Scree Plot Elbow Method:** Plot eigenvalues against component indexes and choose $k$ at the point where the curve flattens out (the "elbow").

**Q4: Explain the difference between Feature Selection and Feature Extraction.**  
**Answer:** Feature Selection keeps a subset of the original features as-is without changing them (e.g. keeping 3 out of 10 columns). Feature Extraction projects the original features into a new, lower-dimensional space, creating entirely new features (e.g. principal components) that are combinations of the original ones.

**Q5: Why is PCA considered an unsupervised algorithm?**  
**Answer:** PCA does not use target class labels to calculate projection directions. It identifies the axes of maximum variance based entirely on the coordinates of the feature data $X$ itself.

---

## Lesson 7.7: References & Further Reading

1.  **Original PCA Paper:** Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space.* Philosophical Magazine, 2(11), 559–572. The foundational paper introducing the geometric concept of PCA.
2.  **Scikit-learn PCA Documentation:** [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). API reference detailing solvers, parameters, and code usage.
3.  **Visual Introduction:** Setosa.io's [Principal Component Analysis Explained Visually](https://setosa.io/ev/principal-component-analysis/). An interactive tool demonstrating how eigenvectors rotate as data points are dragged.
4.  **StatQuest Video Series:** Josh Starmer's [Principal Component Analysis (PCA) Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ). A visual explanation of eigenvalues and eigenvectors.
