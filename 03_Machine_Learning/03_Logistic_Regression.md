# Logistic Regression

## What is Logistic Regression?

Logistic Regression is a statistical model used for binary classification problems (although it can be extended to multi-class classification). Despite its name, it's a classification algorithm, not a regression algorithm. It estimates the probability that a given input belongs to a certain class.

## When to Use Logistic Regression

- When you need a probabilistic framework for binary classification
- When you need interpretable results and understand feature importance
- When you need a fast-training baseline model
- When the relationship between features and the log-odds of the outcome is approximately linear
- When you need the probability of belonging to a class, not just the class prediction
- When working with linearly separable or nearly linearly separable data
- When you need to understand the effect of features on the odds of an outcome

## Mathematical Foundation

### The Logistic Function

At the core of logistic regression is the sigmoid (logistic) function:

σ(z) = 1 / (1 + e^(-z))

Where:

- z is the linear combination of input features: z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- σ(z) outputs a value between 0 and 1, which can be interpreted as a probability

### Model Representation

1. **Linear Predictor**: z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
2. **Probability Transformation**: P(y=1|x) = σ(z) = 1 / (1 + e^(-z))
3. **Decision Rule**: Predict class 1 if P(y=1|x) ≥ threshold (typically 0.5), otherwise class 0

### Log-Odds (Logit) Interpretation

The logistic regression model can be rewritten in terms of log-odds:

log(P(y=1|x) / (1-P(y=1|x))) = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

This representation shows that:

- The log-odds (logit) is a linear function of the features
- Each coefficient βᵢ represents the change in log-odds for a one-unit increase in feature xᵢ
- Exponentiating a coefficient gives the odds ratio for that feature

## How to Implement Logistic Regression

### Basic Implementation in Python with scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
# X: feature matrix, y: target vector (binary)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression(C=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Examine coefficients
coef = pd.DataFrame(
    {'Feature': X.columns,
     'Coefficient': model.coef_[0],
     'Odds Ratio': np.exp(model.coef_[0])
    }).sort_values('Coefficient', ascending=False)
print("\nFeature Coefficients:")
print(coef)
```

### Implementing with Regularization

```python
# L1 Regularization (Lasso) - Encourages sparsity
model_l1 = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42)
model_l1.fit(X_train_scaled, y_train)

# L2 Regularization (Ridge) - Handles multicollinearity
model_l2 = LogisticRegression(penalty='l2', C=0.1, random_state=42)
model_l2.fit(X_train_scaled, y_train)

# Compare coefficients
coef_comparison = pd.DataFrame({
    'Feature': X.columns,
    'No Regularization': model.coef_[0],
    'L1 Regularization': model_l1.coef_[0],
    'L2 Regularization': model_l2.coef_[0]
})
print("\nCoefficient Comparison:")
print(coef_comparison)
```

### Visualizing Decision Boundary (for 2D data)

```python
def plot_decision_boundary(X, y, model, scaler=None):
    # Set min and max values with some margin
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Create features for prediction
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Scale features if a scaler is provided
    if scaler:
        grid = scaler.transform(grid)

    # Make predictions
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

# Example usage:
plot_decision_boundary(X, y, model, scaler)
```

### ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
```

## Types of Logistic Regression

### 1. Binary Logistic Regression

- **When to use**: For problems with two classes (0/1, yes/no, true/false)
- **How it works**: Directly models the probability of an observation belonging to class 1
- **Example applications**:
  - Email spam detection (spam/not spam)
  - Disease diagnosis (present/absent)
  - Customer churn prediction (will churn/won't churn)

### 2. Multinomial Logistic Regression

- **When to use**: For problems with more than two unordered classes
- **How it works**: Uses the softmax function to model probabilities across multiple classes
- **Implementation**:
  ```python
  # Multi-class logistic regression
  multi_class_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
  multi_class_model.fit(X_train_scaled, y_train_multiclass)
  ```
- **Example applications**:
  - Document classification (multiple topics)
  - Product categorization
  - Image classification

### 3. Ordinal Logistic Regression

- **When to use**: For problems with multiple ordered categories
- **How it works**: Models cumulative probabilities using multiple thresholds
- **Implementation**: Not directly available in scikit-learn; requires specialized libraries
- **Example applications**:
  - Survey responses (strongly disagree to strongly agree)
  - Credit ratings (AAA, AA, A, etc.)
  - Education levels (elementary, high school, bachelor's, master's, etc.)

## Key Hyperparameters and Their Tuning

### 1. Regularization Strength (C)

- **What it does**: Controls the penalty on large coefficients
- **How to tune**: Smaller C values increase regularization
- **When to adjust**: When dealing with multicollinearity or overfitting
- **Example**:

  ```python
  from sklearn.model_selection import GridSearchCV

  param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
  grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)
  grid.fit(X_train_scaled, y_train)
  print(f"Best C: {grid.best_params_['C']}")
  ```

### 2. Penalty Type (L1 vs L2)

- **L1 (Lasso)**:
  - Encourages sparse solutions (some coefficients become exactly zero)
  - Good for feature selection
  - When to use: High-dimensional data with many irrelevant features
- **L2 (Ridge)**:
  - Shrinks coefficients toward zero but rarely to exactly zero
  - Better for handling multicollinearity
  - When to use: Features are correlated

### 3. Class Weights

- **What it does**: Adjusts the importance of classes
- **How to tune**: Set to 'balanced' or provide a custom dictionary
- **When to adjust**: For imbalanced datasets
- **Example**:

  ```python
  # Automatic balancing
  model_balanced = LogisticRegression(class_weight='balanced')

  # Custom weights
  model_custom = LogisticRegression(class_weight={0: 1, 1: 5})  # Class 1 is 5x more important
  ```

### 4. Solver Algorithm

- **liblinear**: Good for small datasets, supports L1 regularization
- **lbfgs**: Default, handles multinomial loss, only L2 regularization
- **saga**: Efficient for large datasets, supports both L1 and L2
- **newton-cg**: For multinomial loss and L2 regularization
- **When to adjust**: Based on dataset size, regularization needs, and convergence issues

## Advantages of Logistic Regression

1. **Interpretability**: Coefficients have clear statistical interpretation
2. **Probabilistic Output**: Provides probability estimates, not just classifications
3. **Efficiency**: Fast to train and make predictions
4. **Handles High-Dimensional Data**: Works well even with many features (with regularization)
5. **No Assumptions about Feature Distributions**: Unlike discriminant analysis, doesn't assume normal distributions
6. **Works Well with Sparse Data**: Especially with L1 regularization
7. **Easy to Implement and Update**: Simple to add new training data or features

## Limitations and Challenges

1. **Linearity Assumption**: Assumes linear relationship between features and log-odds

   - **Solution**: Feature engineering to capture non-linearities (polynomial features, interaction terms)

2. **Feature Independence**: Performs best when features are independent

   - **Solution**: Use L2 regularization for correlated features, or perform PCA before modeling

3. **Limited Expressiveness**: Cannot capture complex relationships without feature engineering

   - **Solution**: Consider more flexible models (random forests, neural networks) for complex data

4. **Outlier Sensitivity**: Can be significantly affected by outliers

   - **Solution**: Robust preprocessing, outlier removal, or winsorization

5. **Imbalanced Data Challenges**: May be biased toward the majority class
   - **Solution**: Class weighting, resampling techniques (SMOTE), or adjusted thresholds

## Advanced Topics

### 1. Feature Engineering for Logistic Regression

- **Polynomial Features**: Capture non-linear relationships

  ```python
  from sklearn.preprocessing import PolynomialFeatures

  poly = PolynomialFeatures(degree=2, include_bias=False)
  X_poly = poly.fit_transform(X)
  ```

- **Interaction Terms**: Model how features work together

  ```python
  # Manual interaction
  X['feature1_x_feature2'] = X['feature1'] * X['feature2']
  ```

- **Binning Continuous Variables**: Create threshold effects

  ```python
  from sklearn.preprocessing import KBinsDiscretizer

  discretizer = KBinsDiscretizer(n_bins=5, encode='onehot-dense')
  X_binned = discretizer.fit_transform(X[['continuous_feature']])
  ```

### 2. Handling Imbalanced Data

- **Resampling**:

  ```python
  from imblearn.over_sampling import SMOTE
  from imblearn.under_sampling import RandomUnderSampler

  # Oversampling
  smote = SMOTE(random_state=42)
  X_resampled, y_resampled = smote.fit_resample(X, y)

  # Undersampling
  rus = RandomUnderSampler(random_state=42)
  X_resampled, y_resampled = rus.fit_resample(X, y)
  ```

- **Threshold Adjustment**:

  ```python
  # Find optimal threshold using precision-recall curve
  from sklearn.metrics import precision_recall_curve

  precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

  # F1 score for each threshold
  f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
  optimal_idx = np.argmax(f1_scores)
  optimal_threshold = thresholds[optimal_idx]

  # Apply optimal threshold
  y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
  ```

### 3. Model Interpretation and Explainability

- **Odds Ratios**: Exponentiated coefficients show how odds change with features

  ```python
  odds_ratios = np.exp(model.coef_[0])
  feature_odds = pd.DataFrame({
      'Feature': X.columns,
      'Odds Ratio': odds_ratios
  }).sort_values('Odds Ratio', ascending=False)
  ```

- **Marginal Effects**: How probability changes with feature values

  ```python
  def compute_marginal_effects(model, X, feature_idx):
      # Get current probabilities
      current_probs = model.predict_proba(X)[:, 1]

      # Create copy of X with feature increased by 1 unit
      X_new = X.copy()
      X_new[:, feature_idx] += 1

      # Get new probabilities
      new_probs = model.predict_proba(X_new)[:, 1]

      # Calculate average marginal effect
      return (new_probs - current_probs).mean()

  # Calculate for each feature
  for i, feature in enumerate(X.columns):
      effect = compute_marginal_effects(model, X_scaled, i)
      print(f"Average marginal effect of {feature}: {effect:.4f}")
  ```

- **SHAP Values**: For detailed local explanations

  ```python
  import shap

  explainer = shap.LinearExplainer(model, X_train_scaled)
  shap_values = explainer.shap_values(X_test_scaled)

  # Summary plot
  shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)

  # Force plot for a single prediction
  shap.force_plot(explainer.expected_value, shap_values[0], X_test_scaled[0], feature_names=X.columns)
  ```

## Real-World Applications

### 1. Credit Scoring and Risk Assessment

- **Use case**: Predicting probability of loan default
- **Key features**: Income, credit history, debt-to-income ratio, employment history
- **Implementation considerations**:
  - Class imbalance (defaults are rare)
  - Interpretability requirements for regulatory compliance
  - Calibrated probabilities for risk-based pricing

### 2. Medical Diagnosis

- **Use case**: Predicting disease likelihood based on symptoms and test results
- **Key features**: Lab values, patient demographics, symptoms, medical history
- **Implementation considerations**:
  - Feature engineering for complex medical relationships
  - Cost-sensitive learning (false negatives more costly than false positives)
  - Handling missing data in medical records

### 3. Marketing Campaign Optimization

- **Use case**: Predicting customer response to marketing campaigns
- **Key features**: Demographics, purchase history, browsing behavior, campaign attributes
- **Implementation considerations**:
  - Uplift modeling (targeting customers with highest incremental response)
  - Real-time scoring for online campaigns
  - Integration with business metrics (ROI, LTV)

### 4. Churn Prediction

- **Use case**: Identifying customers likely to cancel service
- **Key features**: Usage patterns, customer service interactions, billing history, competitor actions
- **Implementation considerations**:
  - Time-based features (recent activity vs. historical patterns)
  - Early warning indicators
  - Actionable insights for retention strategies

## Best Practices and Tips

1. **Data Preprocessing**

   - Scale numeric features (StandardScaler or MinMaxScaler)
   - Handle categorical features appropriately (one-hot encoding or embedding)
   - Address missing values (imputation or indicators)
   - Check for and handle outliers

2. **Feature Selection and Engineering**

   - Start with domain knowledge to select relevant features
   - Use L1 regularization for automatic feature selection
   - Create interaction terms for related features
   - Consider non-linear transformations for continuous variables

3. **Model Evaluation**

   - Use appropriate metrics (beyond accuracy)
   - For imbalanced data: precision, recall, F1-score, ROC-AUC
   - For probabilistic evaluation: Brier score, calibration curves
   - Perform cross-validation for reliable performance estimates

4. **Threshold Selection**

   - Don't default to 0.5 threshold
   - Select based on business requirements (cost of false positives vs. false negatives)
   - Consider ROC or precision-recall curves to find optimal threshold

5. **Model Deployment and Monitoring**
   - Save the scaler/preprocessor along with the model
   - Monitor for concept drift over time
   - Retrain periodically with fresh data
   - A/B test model changes before full deployment

## Learning Resources

### Books

- "Applied Logistic Regression" by Hosmer and Lemeshow
- "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani
- "Categorical Data Analysis" by Alan Agresti

### Online Courses

- Stanford's Statistical Learning course
- Coursera: Machine Learning (Andrew Ng)
- edX: Data Science: Machine Learning

### Tools and Libraries

- scikit-learn: Primary implementation for Python
- statsmodels: For statistical details and diagnostics
- R's glm function: For statistical analysis
- SHAP: For model interpretation
- imbalanced-learn: For handling imbalanced datasets
