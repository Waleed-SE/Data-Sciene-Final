# Support Vector Machines (SVM)

## What are Support Vector Machines?

Support Vector Machines (SVMs) are powerful supervised learning algorithms used for classification, regression, and outlier detection. SVMs find the optimal hyperplane that best separates data points of different classes, maximizing the margin between the closest points (support vectors) from each class.

## When to Use SVMs

- When you need a classifier with a clear margin of separation
- When working with high-dimensional data (even when dimensions exceed samples)
- When dealing with structured data rather than unstructured data
- When classes are separable or nearly separable
- When you need a classifier that generalizes well to unseen data
- When you have a small to medium-sized dataset
- When you require non-linear classification using kernel functions
- When you need a method that is robust against overfitting

## Types of SVMs

### 1. Linear SVM

- **When to use**: For linearly separable data or as a first approach
- **How it works**: Finds a hyperplane that maximizes the margin between classes
- **Benefits**: Simple, interpretable, efficient for high-dimensional data
- **Implementation**:

  ```python
  from sklearn.svm import SVC

  # Linear SVM
  linear_svm = SVC(kernel='linear', C=1.0)
  linear_svm.fit(X_train, y_train)
  ```

### 2. Non-linear SVM with Kernels

- **When to use**: When data is not linearly separable
- **How it works**: Uses kernel functions to transform data into higher dimensions where it becomes separable
- **Common kernels**:
  - **Polynomial**: For data with degree-based relationships
  - **Radial Basis Function (RBF)**: For complex non-linear boundaries
  - **Sigmoid**: Similar to neural networks
- **Implementation**:

  ```python
  # RBF Kernel SVM
  rbf_svm = SVC(kernel='rbf', gamma='scale', C=1.0)
  rbf_svm.fit(X_train, y_train)

  # Polynomial Kernel SVM
  poly_svm = SVC(kernel='poly', degree=3, C=1.0)
  poly_svm.fit(X_train, y_train)
  ```

### 3. SVMs for Regression (SVR)

- **When to use**: For regression problems where outliers should have less influence
- **How it works**: Tries to find a function that deviates at most ε from the target
- **Benefits**: Robust to outliers, supports non-linear relationships
- **Implementation**:

  ```python
  from sklearn.svm import SVR

  # SVR with RBF kernel
  svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
  svr.fit(X_train, y_train)
  ```

### 4. One-Class SVM

- **When to use**: For anomaly/outlier detection, when you have mostly normal data
- **How it works**: Learns a boundary that encompasses normal data points
- **Benefits**: Can identify outliers without labeled anomaly examples
- **Implementation**:

  ```python
  from sklearn.svm import OneClassSVM

  # One-Class SVM for anomaly detection
  one_class_svm = OneClassSVM(kernel='rbf', nu=0.1)
  one_class_svm.fit(X_normal)

  # Predict: 1 for inliers, -1 for outliers
  predictions = one_class_svm.predict(X_test)
  ```

## Key Parameters and How to Tune Them

### 1. C (Regularization Parameter)

- **What it controls**: The trade-off between maximizing the margin and minimizing the classification error
- **When to adjust**: To control overfitting/underfitting
- **How to tune**:
  - Low C: Wider margin, more misclassifications allowed (high bias, low variance)
  - High C: Narrower margin, fewer misclassifications (low bias, high variance)
- **Typical range**: 0.1 to 100
- **Example**:

  ```python
  from sklearn.model_selection import GridSearchCV

  param_grid = {'C': [0.1, 1, 10, 100]}
  grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
  grid_search.fit(X_train, y_train)
  best_C = grid_search.best_params_['C']
  ```

### 2. Kernel

- **What it controls**: The type of boundary/decision surface
- **When to adjust**: Based on data's separability characteristics
- **Options**:
  - **'linear'**: For linearly separable data
  - **'poly'**: For data with polynomial relationships
  - **'rbf'**: For complex, non-linear data (default and most versatile)
  - **'sigmoid'**: For neural network-like behavior
  - **Custom kernels**: For specialized domain knowledge
- **How to choose**: Start with RBF, then try others based on performance

### 3. Gamma (RBF Kernel Parameter)

- **What it controls**: The influence range of a single training example
- **When to adjust**: To control the complexity of the decision boundary
- **How to tune**:
  - Low gamma: Smoother, more generalized boundary (higher bias)
  - High gamma: More complex, localized boundary (higher variance)
- **Typical values**: 'scale' (default), 'auto', or specific values like 0.001 to 10
- **Example**:
  ```python
  param_grid = {'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]}
  grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
  grid_search.fit(X_train, y_train)
  best_gamma = grid_search.best_params_['gamma']
  ```

### 4. Degree (Polynomial Kernel Parameter)

- **What it controls**: The degree of the polynomial kernel
- **When to adjust**: When using polynomial kernel to capture non-linear relationships
- **How to tune**: Higher degree = more complex boundary
- **Typical values**: 2, 3, 4 (3 is default)
- **Caution**: Higher degrees can lead to overfitting and numerical issues

### 5. Class Weight

- **What it controls**: The importance of each class
- **When to adjust**: For imbalanced datasets
- **Options**: 'balanced', dictionary of weights, or None
- **Example**:

  ```python
  # Automatically adjust weights inversely proportional to class frequencies
  svm_balanced = SVC(kernel='rbf', class_weight='balanced')

  # Custom weights
  svm_custom = SVC(kernel='rbf', class_weight={0: 1, 1: 10})  # Class 1 is 10x more important
  ```

## Implementation in Python

### Complete SVM Pipeline with scikit-learn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Load and split data
# X: features, y: target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # SVM is sensitive to feature scaling
    ('svm', SVC(probability=True))
])

# Define parameter grid
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01],
    'svm__degree': [2, 3]  # Only relevant for poly kernel
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate on test set
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Get probabilities for ROC curve
y_prob = best_model.predict_proba(X_test)[:, 1]

# Plot ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

### Visualizing SVM Decision Boundaries (2D data)

```python
def plot_decision_boundary(X, y, model, ax=None):
    if ax is None:
        ax = plt.gca()

    # Plot the data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu', edgecolors='k', alpha=0.7)

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Get predictions for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and margins
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.contour(xx, yy, Z, colors='k', linestyles='-', linewidths=1)

    # If SVC model with 'linear' kernel, plot support vectors
    if hasattr(model, 'support_vectors_'):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='k', linewidths=2)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    return ax

# Example usage for different kernels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Linear kernel
svm_linear = SVC(kernel='linear', C=1.0).fit(X, y)
plot_decision_boundary(X, y, svm_linear, ax=axes[0])
axes[0].set_title('Linear Kernel')

# RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X, y)
plot_decision_boundary(X, y, svm_rbf, ax=axes[1])
axes[1].set_title('RBF Kernel')

# Polynomial kernel
svm_poly = SVC(kernel='poly', C=1.0, degree=3).fit(X, y)
plot_decision_boundary(X, y, svm_poly, ax=axes[2])
axes[2].set_title('Polynomial Kernel (degree=3)')

plt.tight_layout()
plt.show()
```

## Mathematical Foundation

### Linear SVM

The optimization problem for a linear SVM is:

minimize: (1/2) ||w||² + C \* Σ ξᵢ

subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ and ξᵢ ≥ 0 for all i

Where:

- w is the normal vector to the hyperplane
- b is the bias term
- ξᵢ are slack variables allowing for misclassifications
- C is the regularization parameter
- (xᵢ, yᵢ) are the training examples and labels

The decision function is: f(x) = sign(w·x + b)

### Kernel SVM

The kernel trick allows SVMs to operate in high-dimensional spaces without explicitly computing the transformation:

K(x, x') = ϕ(x)·ϕ(x')

Common kernel functions:

- Linear: K(x, x') = x·x'
- Polynomial: K(x, x') = (γx·x' + r)^d
- RBF: K(x, x') = exp(-γ||x-x'||²)
- Sigmoid: K(x, x') = tanh(γx·x' + r)

## Advantages of SVMs

1. **Effective in High-Dimensional Spaces**: Particularly suitable when dimensions exceed samples
2. **Memory Efficient**: Only uses a subset of training points (support vectors)
3. **Versatile**: Different kernel functions for various decision boundaries
4. **Robust**: Maximizing the margin helps generalize well to unseen data
5. **Theoretically Well-Founded**: Based on statistical learning theory with strong guarantees
6. **Resistant to Overfitting**: Especially with proper regularization
7. **Handles Non-linear Problems**: Through kernel functions

## Limitations and Challenges

1. **Computational Complexity**: O(n²) to O(n³) scaling with dataset size

   - **Solution**: Use LinearSVC for large datasets or consider SGDClassifier with hinge loss

2. **Sensitivity to Feature Scaling**: Features should be on similar scales

   - **Solution**: Always standardize/normalize features before training

3. **Hyperparameter Selection**: Performance heavily depends on C, gamma, kernel choice

   - **Solution**: Systematic grid search or Bayesian optimization for tuning

4. **Not Directly Probabilistic**: Standard SVMs don't provide probability estimates

   - **Solution**: Use `probability=True` option with Platt scaling (but increases training time)

5. **Challenging for Large Datasets**: Training time can be prohibitive for millions of samples

   - **Solution**: Consider linear kernels, stochastic gradient methods, or subsampling

6. **Black Box Nature**: Less interpretable than simpler models
   - **Solution**: For linear kernels, examine coefficients; for non-linear, analyze support vectors

## Real-World Applications

### 1. Text Classification

- **When to use**: For document categorization, sentiment analysis
- **How to implement**: Convert text to vectors (TF-IDF, word embeddings), then apply SVM
- **Benefits**: Works well with high-dimensional sparse data
- **Example**:

  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.pipeline import Pipeline
  from sklearn.svm import SVC

  text_clf = Pipeline([
      ('tfidf', TfidfVectorizer()),
      ('clf', SVC(kernel='linear'))
  ])
  text_clf.fit(X_train_texts, y_train)
  ```

### 2. Image Classification

- **When to use**: For small to medium image datasets, especially with extracted features
- **How to implement**: Extract features (HOG, SIFT) or use CNN embeddings, then apply SVM
- **Benefits**: Can perform well with fewer examples than deep learning
- **Example with HOG features**:

  ```python
  from skimage.feature import hog
  from sklearn.svm import SVC

  def extract_hog_features(images):
      features = []
      for image in images:
          hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
          features.append(hog_features)
      return np.array(features)

  X_train_hog = extract_hog_features(X_train_images)
  X_test_hog = extract_hog_features(X_test_images)

  svm = SVC(kernel='rbf')
  svm.fit(X_train_hog, y_train)
  ```

### 3. Bioinformatics

- **When to use**: For protein classification, gene expression analysis, etc.
- **How to implement**: Custom kernels designed for biological sequences
- **Benefits**: Can handle structured biological data effectively
- **Example of string kernel for protein sequences**:

  ```python
  def string_kernel(X, Y, k=3):
      """Simple k-mer string kernel for sequences"""
      result = np.zeros((len(X), len(Y)))

      for i, x in enumerate(X):
          for j, y in enumerate(Y):
              x_kmers = [x[l:l+k] for l in range(len(x)-k+1)]
              y_kmers = [y[l:l+k] for l in range(len(y)-k+1)]

              # Count common k-mers
              common = set(x_kmers).intersection(set(y_kmers))
              result[i,j] = len(common)

      return result

  # Using custom kernel
  from sklearn.svm import SVC
  svm = SVC(kernel=string_kernel)
  svm.fit(protein_sequences, labels)
  ```

### 4. Anomaly Detection

- **When to use**: For fraud detection, network intrusion, manufacturing defects
- **How to implement**: One-Class SVM trained on normal examples
- **Benefits**: Can detect outliers without labeled anomaly examples
- **Example**:

  ```python
  from sklearn.svm import OneClassSVM

  # Train on normal data only
  one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
  one_class_svm.fit(X_normal)

  # Predict: 1 for inliers, -1 for outliers
  y_pred = one_class_svm.predict(X_test)
  anomalies = X_test[y_pred == -1]
  ```

## Best Practices

### 1. Data Preprocessing

- **Feature Scaling**: Always standardize features (mean=0, std=1)

  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  ```

- **Handling Missing Values**: Impute missing values before training

  ```python
  from sklearn.impute import SimpleImputer
  imputer = SimpleImputer(strategy='mean')
  X_train_imputed = imputer.fit_transform(X_train)
  ```

- **Dimensionality Reduction**: Consider PCA for very high-dimensional data
  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=0.95)  # Retain 95% of variance
  X_train_reduced = pca.fit_transform(X_train_scaled)
  ```

### 2. Model Selection and Tuning

- **Start Simple**: Try linear kernel first, then move to non-linear if needed
- **Systematic Hyperparameter Tuning**: Use grid search or randomized search

  ```python
  from sklearn.model_selection import RandomizedSearchCV
  from scipy.stats import uniform, loguniform

  param_dist = {
      'C': loguniform(1e-3, 1e3),
      'gamma': loguniform(1e-4, 1e0),
      'kernel': ['rbf', 'poly']
  }

  random_search = RandomizedSearchCV(
      SVC(), param_dist, n_iter=100, cv=5, n_jobs=-1
  )
  random_search.fit(X_train_scaled, y_train)
  ```

- **Cross-Validation**: Always use k-fold CV for reliable performance estimates
- **Consider Multiple Metrics**: Accuracy, F1, AUC depending on problem

### 3. Handling Imbalanced Data

- **Class Weights**: Use `class_weight='balanced'` or custom weights
- **Sampling Techniques**: Apply SMOTE or other resampling methods

  ```python
  from imblearn.over_sampling import SMOTE

  smote = SMOTE(random_state=42)
  X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
  ```

### 4. Making SVMs Scalable

- **For Large Datasets**:

  - Use LinearSVC (faster implementation for linear kernels)

    ```python
    from sklearn.svm import LinearSVC

    linear_svc = LinearSVC(dual=False, C=1.0)
    linear_svc.fit(X_train_scaled, y_train)
    ```

  - Consider SGDClassifier with hinge loss (approximates SVM)

    ```python
    from sklearn.linear_model import SGDClassifier

    sgd_svm = SGDClassifier(loss='hinge', alpha=1/C)
    sgd_svm.fit(X_train_scaled, y_train)
    ```

  - Use subsampling or online learning for very large datasets

## Resources for Learning More

### Books and Papers

- "Support Vector Machines" by Cristianini and Shawe-Taylor
- "Learning with Kernels" by Schölkopf and Smola
- The original SVM paper: "Support-Vector Networks" by Cortes and Vapnik

### Online Courses and Tutorials

- Stanford's CS229 (Machine Learning) SVM lectures by Andrew Ng
- scikit-learn SVM documentation and examples
- "Understanding Support Vector Machine Regression" on the MathWorks blog

### Advanced Topics

- Multiple Kernel Learning
- Structured SVMs for complex outputs
- Online SVMs for streaming data
- Twin SVMs and multi-class optimization strategies
