# Singular Value Decomposition (SVD) and Principal Component Analysis (PCA)

## What are SVD and PCA?

**Singular Value Decomposition (SVD)** is a matrix factorization technique that decomposes a matrix into three other matrices. Mathematically, for a matrix A, SVD gives: A = UΣV^T, where U and V are orthogonal matrices and Σ is a diagonal matrix of singular values.

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms the data into a new coordinate system where the greatest variances lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

## Relationship Between SVD and PCA

PCA can be computed using SVD. When applied to a centered data matrix X:

- The right singular vectors (V) are the principal components
- The singular values (Σ) are related to the explained variance
- The left singular vectors (U) multiplied by Σ give the projections of the data onto the principal components

## When to Use SVD and PCA

### SVD Use Cases

- When you need a general matrix decomposition tool
- For solving systems of linear equations
- For computing pseudo-inverses of matrices
- In image compression and noise reduction
- As a component in recommendation systems
- When working with non-square matrices
- For numerical stability in computational problems

### PCA Use Cases

- When reducing dimensionality while preserving variance
- When visualizing high-dimensional data
- When removing multicollinearity before regression
- For feature extraction and noise filtering
- When compressing data with minimal information loss
- As a preprocessing step for machine learning algorithms
- When uncovering latent structures in the data

## How to Use SVD

### Mathematical Formulation

For a matrix A of size m×n, SVD decomposes it as:
A = UΣV^T where:

- U is an m×m orthogonal matrix containing left singular vectors
- Σ is an m×n diagonal matrix containing singular values σ₁ ≥ σ₂ ≥ ... ≥ 0
- V^T is the transpose of an n×n orthogonal matrix containing right singular vectors

### Implementation in Python

```python
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Full SVD using NumPy
A = np.array([[1, 2], [3, 4], [5, 6]])
U, sigma, VT = np.linalg.svd(A, full_matrices=False)
# Reconstruction
A_reconstructed = U @ np.diag(sigma) @ VT

# Truncated SVD using scikit-learn (for large matrices)
svd = TruncatedSVD(n_components=2)
A_transformed = svd.fit_transform(A)
```

### Key Applications of SVD

#### 1. Low-rank Approximation

- **When to use**: For data compression or noise reduction
- **How to use**: Keep only the k largest singular values and corresponding vectors
- **Benefits**: Optimal approximation (Eckart-Young theorem)
- **Example**:
  ```python
  # Low-rank approximation with k components
  k = 1
  A_approx = U[:, :k] @ np.diag(sigma[:k]) @ VT[:k, :]
  ```

#### 2. Pseudo-inverse Calculation

- **When to use**: For solving linear systems where the matrix may be singular
- **How to use**: A⁺ = V Σ⁺ U^T where Σ⁺ contains reciprocals of non-zero singular values
- **Benefits**: Numerically stable solution to ill-conditioned systems
- **Example**:
  ```python
  # Pseudo-inverse using SVD
  sigma_plus = np.zeros(A.shape[::-1])
  for i in range(len(sigma)):
      sigma_plus[i, i] = 1/sigma[i] if sigma[i] > 1e-10 else 0
  A_plus = VT.T @ sigma_plus @ U.T
  ```

#### 3. Recommendation Systems

- **When to use**: For collaborative filtering in recommender systems
- **How to use**: Decompose user-item interaction matrix and use factors for predictions
- **Benefits**: Captures latent factors in user preferences
- **Example**:
  ```python
  # Simplified matrix factorization for recommendations
  user_item_matrix = np.array([[5, 3, 0, 1],
                              [4, 0, 0, 1],
                              [1, 1, 0, 5]])
  U, sigma, VT = np.linalg.svd(user_item_matrix, full_matrices=False)
  k = 2  # Number of latent factors
  user_factors = U[:, :k] @ np.diag(np.sqrt(sigma[:k]))
  item_factors = np.diag(np.sqrt(sigma[:k])) @ VT[:k, :]
  # Predict ratings
  predicted_ratings = user_factors @ item_factors
  ```

## How to Use PCA

### Mathematical Formulation

1. Center the data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors and eigenvalues of covariance matrix
4. Sort eigenvectors by decreasing eigenvalues
5. Project data onto eigenvectors (principal components)

### Implementation in Python

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Using scikit-learn
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")

# Manual PCA implementation
X_centered = X - X.mean(axis=0)
cov_matrix = np.cov(X_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# Sort by decreasing eigenvalues
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]
eigenvalues = eigenvalues[idx]
# Project data
X_pca_manual = X_centered @ eigenvectors
```

### Key Applications of PCA

#### 1. Dimensionality Reduction

- **When to use**: When dealing with high-dimensional data
- **How to use**: Project data onto k principal components
- **Benefits**: Preserves maximum variance in reduced dimensions
- **Example**:

  ```python
  # Reduce to 2 dimensions
  pca = PCA(n_components=2)
  X_reduced = pca.fit_transform(X)

  # Visualize
  plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
  plt.xlabel('First Principal Component')
  plt.ylabel('Second Principal Component')
  plt.title('PCA Visualization')
  plt.show()
  ```

#### 2. Feature Extraction

- **When to use**: To create uncorrelated features for machine learning
- **How to use**: Use principal components as new features
- **Benefits**: Removes multicollinearity, often improves model performance
- **Example**:

  ```python
  # Use PCA for feature extraction before training a model
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.pipeline import Pipeline

  pipeline = Pipeline([
      ('pca', PCA(n_components=10)),
      ('rf', RandomForestClassifier(random_state=42))
  ])
  pipeline.fit(X_train, y_train)
  ```

#### 3. Noise Reduction

- **When to use**: When data contains noise in minor components
- **How to use**: Project to principal components and reconstruct using only top k
- **Benefits**: Filters out noise while preserving signal
- **Example**:
  ```python
  # Noise reduction
  pca = PCA(n_components=5)  # Keep top 5 components
  X_reduced = pca.fit_transform(X_noisy)
  X_reconstructed = pca.inverse_transform(X_reduced)
  ```

#### 4. Visualization

- **When to use**: To visualize high-dimensional data in 2D or 3D
- **How to use**: Project data onto first 2 or 3 principal components
- **Benefits**: Preserves maximum variance for visual interpretation
- **Example**: See dimensionality reduction example above

## Best Practices and Considerations

### 1. Data Preprocessing

- **Scaling**: Always standardize (zero mean, unit variance) before PCA
- **Missing Values**: Handle missing values before applying PCA/SVD
- **Outliers**: Consider removing or transforming outliers as they can distort results

### 2. Selecting Number of Components

- **Scree Plot**: Plot explained variance vs. number of components
- **Cumulative Variance**: Select k components that explain sufficient variance (e.g., 95%)
- **Kaiser's Rule**: Keep components with eigenvalues > 1
- **Example**:

  ```python
  # Scree plot
  pca = PCA()
  pca.fit(X_standardized)

  plt.figure(figsize=(10, 6))
  plt.plot(np.cumsum(pca.explained_variance_ratio_))
  plt.xlabel('Number of Components')
  plt.ylabel('Cumulative Explained Variance')
  plt.axhline(y=0.95, color='r', linestyle='-')
  plt.title('Explained Variance vs. Components')
  plt.grid(True)
  plt.show()

  # Find number of components for 95% variance
  n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
  print(f"Number of components for 95% variance: {n_components}")
  ```

### 3. Interpretation

- **Component Loadings**: Examine the weights of original features in components
- **Biplot**: Visualize both observations and feature loadings
- **Example**:

  ```python
  # Plot feature loadings
  plt.figure(figsize=(12, 8))
  for i, feature in enumerate(feature_names):
      plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], head_width=0.05)
      plt.text(pca.components_[0, i]*1.2, pca.components_[1, i]*1.2, feature)

  plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
  plt.xlabel('PC1')
  plt.ylabel('PC2')
  plt.grid()
  plt.title('PCA Biplot')
  plt.show()
  ```

### 4. Scalability Considerations

- **Memory Usage**: Full SVD/PCA can be memory-intensive for large matrices
- **Computational Efficiency**: Use randomized or incremental algorithms for large datasets
- **Example**:

  ```python
  # Randomized SVD for large matrices
  from sklearn.decomposition import TruncatedSVD

  svd = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=5)
  X_reduced = svd.fit_transform(large_sparse_matrix)
  ```

### 5. Nonlinear Alternatives

- Consider nonlinear dimensionality reduction (Kernel PCA, t-SNE, UMAP) when data has nonlinear structure

## Differences Between SVD and PCA

| Aspect                       | SVD                          | PCA                                         |
| ---------------------------- | ---------------------------- | ------------------------------------------- |
| Input Matrix                 | Any matrix                   | Typically covariance/correlation matrix     |
| Centered Data                | Not required                 | Required (mean subtraction)                 |
| Output                       | Three matrices (U, Σ, V^T)   | Loadings and scores                         |
| Application Focus            | General matrix factorization | Variance-based dimensionality reduction     |
| Handling Non-square Matrices | Natural                      | Requires additional steps                   |
| Computational Efficiency     | Can be more efficient        | May require explicit covariance computation |

## Limitations and Challenges

### SVD Limitations

- Computationally expensive for large matrices
- Not ideal for sparse matrices in naive implementation
- Challenging to update incrementally with new data

### PCA Limitations

- Assumes linear relationships in data
- Sensitive to scaling of input features
- May not preserve important information if variance doesn't align with importance
- Cannot handle categorical features directly

## Advanced Topics

### 1. Kernel PCA

- **When to use**: When data has nonlinear patterns
- **How it works**: Applies kernel trick to perform PCA in high-dimensional feature space
- **Example**:

  ```python
  from sklearn.decomposition import KernelPCA

  kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
  X_kpca = kpca.fit_transform(X)
  ```

### 2. Sparse PCA and SVD

- **When to use**: When sparse loadings are desired for interpretability
- **How it works**: Adds sparsity constraints to component loadings
- **Example**:

  ```python
  from sklearn.decomposition import SparsePCA

  sparse_pca = SparsePCA(n_components=5, alpha=1)
  X_sparse_pca = sparse_pca.fit_transform(X)
  ```

### 3. Incremental/Online PCA

- **When to use**: For streaming data or very large datasets
- **How it works**: Updates PCA model incrementally with new observations
- **Example**:

  ```python
  from sklearn.decomposition import IncrementalPCA

  ipca = IncrementalPCA(n_components=10, batch_size=100)
  for batch in data_batches:
      ipca.partial_fit(batch)
  ```

### 4. Robust PCA

- **When to use**: When data contains outliers
- **How it works**: Decomposes matrix into low-rank and sparse components
- **Benefits**: Separates structured noise from signal

## Resources for Further Learning

### Books

- "Introduction to Linear Algebra" by Gilbert Strang
- "Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

### Online Courses

- MIT OpenCourseWare: Linear Algebra by Gilbert Strang
- Coursera: Mathematics for Machine Learning: PCA

### Software Documentation

- Scikit-learn: [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- Scikit-learn: [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- NumPy: [numpy.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
