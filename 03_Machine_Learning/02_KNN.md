# K-Nearest Neighbors (KNN)

## What is K-Nearest Neighbors?

K-Nearest Neighbors (KNN) is a simple, instance-based machine learning algorithm used for classification and regression tasks. Unlike parametric methods that learn a model, KNN memorizes the training dataset and makes predictions based on the similarity (distance) between new data points and stored examples.

## When to Use KNN

- When the relationship between features and target is complex or unknown
- When the dataset is small to moderate in size
- When data has low dimensionality (or after dimensionality reduction)
- For recommendation systems (content-based filtering)
- For anomaly detection
- When interpretability of predictions is important
- As a baseline algorithm to compare against more complex models
- When training speed is more important than prediction speed

## How KNN Works

### Basic Algorithm

1. **Store**: Memorize the entire training dataset
2. **Calculate**: For a new data point, calculate the distance to all training examples
3. **Sort**: Rank the training examples by distance
4. **Vote/Average**:
   - For classification: Take majority vote of the k nearest neighbors
   - For regression: Calculate the average (or weighted average) of the k nearest neighbors
5. **Predict**: Return the voted class or average value

### Distance Metrics

Different distance measures can be used depending on the data type and problem:

1. **Euclidean Distance** (L2 norm)

   - **When to use**: For continuous features, when the scale of individual features matters
   - **Formula**: sqrt(sum((x_i - y_i)²))
   - **Benefits**: Intuitive, works well in low dimensions

2. **Manhattan Distance** (L1 norm)

   - **When to use**: For grid-like features, feature spaces where diagonal movement isn't natural
   - **Formula**: sum(|x_i - y_i|)
   - **Benefits**: Less sensitive to outliers than Euclidean distance

3. **Minkowski Distance**

   - **When to use**: As a generalization of Euclidean and Manhattan distance
   - **Formula**: (sum(|x_i - y_i|^p))^(1/p) where p is a parameter
   - **Benefits**: Flexibility to adjust between Manhattan (p=1) and Euclidean (p=2)

4. **Hamming Distance**

   - **When to use**: For categorical features, comparing strings or binary vectors
   - **Formula**: Count of positions where characters/bits differ
   - **Benefits**: Natural choice for non-numeric data

5. **Cosine Similarity**
   - **When to use**: When the magnitude of vectors is less important than their direction
   - **Formula**: dot(x,y)/(||x||\*||y||)
   - **Benefits**: Good for text data and high-dimensional sparse data

### Choosing K Value

The k value determines how many neighbors influence the prediction:

- **Small k** (e.g., k=1, k=3):

  - More sensitive to local patterns
  - Can capture complex decision boundaries
  - More susceptible to noise and outliers
  - May lead to overfitting

- **Large k** (e.g., k=10, k=20):
  - Smoother decision boundaries
  - More robust to noise
  - May overlook important local patterns
  - Can lead to underfitting

**Best practice**: Use cross-validation to find the optimal k value for your specific dataset.

### Weighted KNN

In weighted KNN, closer neighbors have more influence on the prediction:

- **Distance-based weights**: Typically 1/distance or 1/distance²
- **When to use**: When closer neighbors are more relevant to the prediction
- **Benefits**: Smoother transitions between classes, more nuanced predictions

## Implementation in Python

### Basic KNN Classification

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Sample data (features and labels)
X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
y = np.array([0, 0, 0, 1, 1, 1])  # Binary classification

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create and train the model
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Predict for a new point
new_point = np.array([[4, 5]])
prediction = knn.predict(new_point)
print(f"Prediction for {new_point}: {prediction}")

# Get probability estimates
probabilities = knn.predict_proba(new_point)
print(f"Probabilities: {probabilities}")
```

### KNN Regression

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([3, 5, 7, 9, 11, 13, 15, 17])  # Linear pattern: y = 2x + 1 with noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create and train the model
k = 3
knn_reg = KNeighborsRegressor(n_neighbors=k)
knn_reg.fit(X_train, y_train)

# Make predictions
y_pred = knn_reg.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Predict for a new point
new_point = np.array([[5.5]])
prediction = knn_reg.predict(new_point)
print(f"Predicted value for {new_point}: {prediction}")
```

### Finding Optimal K with Cross-Validation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Create parameter grid
param_grid = {'n_neighbors': range(1, 21)}

# Setup grid search
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    return_train_score=True
)

# Perform grid search
grid_search.fit(X, y)

# Results
print(f"Best k: {grid_search.best_params_['n_neighbors']}")
print(f"Best accuracy: {grid_search.best_score_:.4f}")

# Plot results
import matplotlib.pyplot as plt

train_scores = grid_search.cv_results_['mean_train_score']
test_scores = grid_search.cv_results_['mean_test_score']
k_values = range(1, 21)

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, label='Training Accuracy')
plt.plot(k_values, test_scores, label='Validation Accuracy')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN Performance vs. k Value')
plt.legend()
plt.grid(True)
plt.show()
```

### Using Different Distance Metrics

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# List of metrics to try
metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
k = 5

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    print(f"{metric.capitalize()} metric - Avg accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Implementing Weighted KNN

```python
from sklearn.neighbors import KNeighborsClassifier

# Uniform weights (standard KNN)
knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# Distance-based weights
knn_distance = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Compare performance
uniform_scores = cross_val_score(knn_uniform, X, y, cv=5, scoring='accuracy')
distance_scores = cross_val_score(knn_distance, X, y, cv=5, scoring='accuracy')

print(f"Uniform weights - Avg accuracy: {uniform_scores.mean():.4f}")
print(f"Distance weights - Avg accuracy: {distance_scores.mean():.4f}")
```

## Advantages of KNN

1. **Simplicity**: Easy to understand and implement
2. **No Training Phase**: Just stores the data (lazy learning)
3. **Naturally Handles Multi-class Problems**: No need for special adaptations
4. **Non-parametric**: Makes no assumptions about the underlying data distribution
5. **Adaptability**: New training data can be added seamlessly
6. **Interpretability**: Predictions can be explained by showing the nearest neighbors

## Limitations and Challenges

1. **Curse of Dimensionality**: Performance degrades in high-dimensional spaces

   - **Solution**: Apply dimensionality reduction (PCA, t-SNE) before KNN

2. **Computational Cost**: Prediction time increases with dataset size

   - **Solution**: Use approximate nearest neighbor algorithms (KD-trees, Ball trees)

3. **Sensitivity to Scale**: Features with larger ranges can dominate distance calculations

   - **Solution**: Standardize or normalize features

4. **Imbalanced Data**: Majority class dominates predictions

   - **Solution**: Use weighted voting, adjust class weights, or balance the dataset

5. **Outliers**: Can significantly affect predictions

   - **Solution**: Use robust distance metrics or outlier detection

6. **Memory Requirements**: Stores the entire training set
   - **Solution**: Data condensation techniques to store representative subset

## Advanced KNN Concepts

### 1. Approximate Nearest Neighbors

For large datasets, exact KNN can be slow. Approximate methods include:

- **KD-Trees**: Space-partitioning data structure

  - **When to use**: Low to moderate dimensions (typically < 20)
  - **How it works**: Recursively partitions space along coordinate axes
  - **Implementation**:

    ```python
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
    knn.fit(X_train, y_train)
    ```

- **Ball Trees**: Hierarchical data structure

  - **When to use**: Higher dimensions than KD-Trees can handle
  - **How it works**: Partitions space into nested hyperspheres
  - **Implementation**:
    ```python
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    knn.fit(X_train, y_train)
    ```

- **Locality-Sensitive Hashing (LSH)**: Hashing technique for approximate similarity search
  - **When to use**: Very high dimensions and large datasets
  - **How it works**: Hashes similar items to the same bucket with high probability

### 2. Distance Metric Learning

Learning a custom distance metric from the data:

- **Mahalanobis Distance**: Accounts for correlations between features
- **Large Margin Nearest Neighbor (LMNN)**: Optimizes distance metric to improve classification
- **Neighborhood Components Analysis (NCA)**: Learns a distance metric that maximizes KNN performance

### 3. Condensed Nearest Neighbors

Techniques to reduce the stored dataset size:

- **Condensed Nearest Neighbor (CNN)**: Stores only points near decision boundaries
- **Edited Nearest Neighbor (ENN)**: Removes points that disagree with their neighbors
- **Instance Selection**: Various algorithms to select representative instances

## Real-World Applications

### 1. Recommendation Systems

- **How KNN is used**: Find similar users or items based on preferences/features
- **Example**: Movie recommendations based on user ratings
- **Implementation approach**: Collaborative filtering using user-user or item-item similarity

### 2. Image Recognition

- **How KNN is used**: Classify images based on similarity to labeled examples
- **Example**: Handwritten digit recognition
- **Implementation approach**: Use feature extraction methods before applying KNN

### 3. Medical Diagnosis

- **How KNN is used**: Classify patients based on similarity to known cases
- **Example**: Predicting disease likelihood based on patient attributes
- **Implementation approach**: Careful feature selection and weighted KNN

### 4. Anomaly Detection

- **How KNN is used**: Identify points that are distant from their neighbors
- **Example**: Fraud detection in credit card transactions
- **Implementation approach**: Calculate average distance to k neighbors, flag outliers

### 5. Missing Value Imputation

- **How KNN is used**: Predict missing values based on similar instances
- **Example**: Filling gaps in sensor data
- **Implementation approach**: Use KNN regression on complete samples to predict missing values

## Best Practices for KNN

1. **Preprocessing**

   - **Feature Scaling**: Always normalize or standardize features
   - **Dimensionality Reduction**: Apply PCA or feature selection for high-dimensional data
   - **Handle Categorical Features**: Use appropriate encoding and distance metrics

2. **Model Selection and Evaluation**

   - **Select k via Cross-Validation**: Typically test odd values to avoid ties
   - **Distance Metric Selection**: Try multiple metrics and select based on cross-validation
   - **Use Stratified Cross-Validation**: Especially for imbalanced datasets

3. **Performance Optimization**

   - **Approximate Algorithms**: Use KD-trees or Ball trees for large datasets
   - **Feature Engineering**: Create domain-specific features
   - **Weighted KNN**: Use distance-based weights for smoother decision boundaries

4. **Implementation Considerations**
   - **Memory Usage**: Be mindful of dataset size
   - **Prediction Time**: Consider using approximate methods for real-time applications
   - **Parallel Processing**: Leverage multi-core processing for distance calculations

## Resources for Learning More

### Libraries and Tools

- Scikit-learn: KNeighborsClassifier, KNeighborsRegressor
- FAISS (Facebook AI Similarity Search): For efficient similarity search
- Annoy (Approximate Nearest Neighbors Oh Yeah): Spotify's library for approximate nearest neighbors

### Books and Articles

- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Machine Learning: A Probabilistic Perspective" by Kevin Murphy
- "Instance-Based Learning" chapter in "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

### Online Courses

- Coursera: Machine Learning (Andrew Ng)
- edX: Data Science and Machine Learning Essentials
- Fast.ai: Practical Machine Learning
