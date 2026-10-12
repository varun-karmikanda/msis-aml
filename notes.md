# Machine Learning Practicals — Quick Revision Guide

A cheat sheet for remembering common Scikit-learn practical implementations.

## 1. Universal ML Workflow

Remember:

**Import → Load → Split → Pipeline → Fit → Predict → Metrics → Cross-validation → Plot**

Most ML practical programs follow this structure. Only the model, metrics, and special requirements change.

## 2. Universal Code Template

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 1. Load dataset
# X = features
# y = target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Create pipeline
model = Pipeline([
    ("model", YOUR_MODEL())
])

# 4. Train
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate using the required metrics
```

Replace `YOUR_MODEL()` and the evaluation section according to the question.

---

## 3. Dataset Loading

### Built-in classification dataset

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
```

### Built-in regression dataset

```python
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
X = data.data
y = data.target
```

### Important variables

| Variable | Meaning |
|---|---|
| `X` | Input features |
| `y` | Target/output |
| `X_train` | Training features |
| `X_test` | Testing features |
| `y_train` | Actual training targets |
| `y_test` | Actual testing targets |
| `y_pred` | Model predictions |

---

## 4. Model Cheat Sheet

Import the appropriate model from its Scikit-learn module.

| Practical | Model |
|---|---|
| Decision Tree Classification | `DecisionTreeClassifier(criterion="entropy")` |
| Decision Tree using Gini | `DecisionTreeClassifier(criterion="gini")` |
| Pre-pruning | `DecisionTreeClassifier(max_depth=4)` |
| Post-pruning | `DecisionTreeClassifier(ccp_alpha=0.005)` |
| Decision Tree Regression | `DecisionTreeRegressor(max_depth=4)` |
| Linear Regression | `LinearRegression()` |
| Logistic Regression — Ridge | `LogisticRegression(penalty="l2", max_iter=5000)` |
| Logistic Regression — Lasso | `LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000)` |

Required imports:

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
```

**Remember:** The values `max_depth=4` and `ccp_alpha=0.005` are examples. They are not guaranteed to be optimal.

---

## 5. Classification Metrics

Used for Decision Tree Classification and Logistic Regression.

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, zero_division=0))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
```

| Metric | Meaning |
|---|---|
| Accuracy | Correct predictions / total predictions |
| Precision | Correct positive predictions / all positive predictions |
| Recall | Correct positive predictions / all actual positives |
| F1-score | Harmonic mean of precision and recall |

For multiclass classification, specify an appropriate averaging method such as `average="weighted"` when calculating precision, recall, and F1.

---

## 6. Regression Metrics

Used for Linear Regression and Decision Tree Regression.

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2-score:", r2)
```

| Metric | Meaning | Preferred result |
|---|---|---|
| MSE | Mean squared prediction error | Lower |
| RMSE | Square root of MSE | Lower |
| MAE | Mean absolute prediction error | Lower |
| R² | Fit relative to predicting the target mean | Generally higher |

**Shortcut:** Classification → Accuracy, Precision, Recall, F1. Regression → MSE, RMSE, R².

---

## 7. Cross-Validation

### Classification: StratifiedKFold

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1"]
)

print("CV Accuracy:", scores["test_accuracy"].mean())
print("CV Precision:", scores["test_precision"].mean())
print("CV Recall:", scores["test_recall"].mean())
print("CV F1:", scores["test_f1"].mean())
```

### Regression: KFold

```python
from sklearn.model_selection import KFold, cross_validate

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=["neg_mean_squared_error", "r2"]
)

cv_mse = -scores["test_neg_mean_squared_error"].mean()
cv_r2 = scores["test_r2"].mean()

print("CV MSE:", cv_mse)
print("CV R2:", cv_r2)
```

**Remember:**
- Classification → `StratifiedKFold`
- Regression → `KFold`
- `cross_validate()` evaluates multiple metrics.
- Use the training data for cross-validation; reserve the test data for final evaluation.
- Fit preprocessing inside the pipeline to avoid data leakage.

---

## 8. Logistic Regression — Special Requirements

### Sigmoid function

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

z = np.linspace(-10, 10, 200)
p = sigmoid(z)

plt.plot(z, p)
plt.axhline(0.5, linestyle="--")
plt.xlabel("z")
plt.ylabel("Probability")
plt.title("Sigmoid Function")
plt.show()
```

Formula:

`sigmoid(z) = 1 / (1 + exp(-z))`

It converts a linear score into a probability between 0 and 1.

### Decision boundary

For binary logistic regression with the default threshold of 0.5:

`P(y = 1 | X) = 0.5`

The boundary separates the two predicted classes.

To visualize it, train a model using two features, create a meshgrid, predict on the grid, and plot the class regions with `plt.contourf()`.

### Ridge and Lasso regularization

```python
# Ridge = L2
LogisticRegression(penalty="l2", C=1.0, max_iter=5000)

# Lasso = L1
LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    max_iter=5000
)
```

- **Ridge/L2:** Shrinks coefficients toward zero.
- **Lasso/L1:** Can set coefficients exactly to zero.
- Smaller `C` means stronger regularization.
- Logistic regression generally benefits from scaling, so use `StandardScaler()` inside its pipeline.

---

## 9. Decision Tree — Special Requirements

### Entropy and Information Gain

Entropy measures uncertainty:

`H(S) = -sum(p_i * log2(p_i))`

Information gain measures the reduction in entropy after a split:

`IG = H(parent) - weighted entropy(children)`

In Scikit-learn:

```python
DecisionTreeClassifier(criterion="entropy")
```

`criterion="entropy"` selects splits using information gain.

### Gini impurity

`Gini(S) = 1 - sum(p_i ** 2)`

```python
DecisionTreeClassifier(criterion="gini")
```

`entropy` and `gini` are different splitting criteria. `log_loss` is also available as an alternative entropy-based criterion.

### Overfitting and pruning

```python
# Unpruned tree
DecisionTreeClassifier()

# Pre-pruning
DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)

# Post-pruning
DecisionTreeClassifier(ccp_alpha=0.005)
```

Compare training accuracy and test accuracy:

```python
print("Train:", model.score(X_train, y_train))
print("Test:", model.score(X_test, y_test))
```

A large training-test performance gap can indicate overfitting.

- **Pre-pruning:** Limits tree growth.
- **Post-pruning:** Removes branches after growing the tree.
- `ccp_alpha` controls the strength of cost-complexity pruning.

---

## 10. Linear Regression — Special Requirements

```python
from sklearn.linear_model import LinearRegression

model = Pipeline([
    ("model", LinearRegression())
])
```

Linear regression predicts a continuous target using a linear relationship:

`y_hat = b0 + b1*x1 + b2*x2 + ... + bn*xn`

Train and evaluate it using the universal workflow and regression metrics.

Standard scaling is not required for ordinary linear regression, although it may be useful for some extensions.

---

## 11. Pipeline Pattern

Use the same structure for all models:

```python
model = Pipeline([
    ("scaler", StandardScaler()),  # If needed
    ("model", YOUR_MODEL())
])
```

For decision trees, the scaler can be omitted:

```python
model = Pipeline([
    ("model", DecisionTreeClassifier())
])
```

For logistic regression, scaling is generally useful:

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

For Linear Regression:

```python
model = Pipeline([
    ("model", LinearRegression())
])
```

---

## 12. Final Exam Memory Table

| If the question mentions... | Remember to add... |
|---|---|
| Classification | Accuracy, precision, recall, F1 |
| Regression | MSE, RMSE, R² |
| Cross-validation | `cross_validate()` and a CV splitter |
| Entropy | `criterion="entropy"` |
| Information gain | Entropy reduction |
| Gini impurity | `criterion="gini"` |
| Overfitting | Compare train and test performance |
| Pre-pruning | `max_depth`, `min_samples_leaf` |
| Post-pruning | `ccp_alpha` |
| Sigmoid | Probability formula and plot |
| Decision boundary | Two features and a meshgrid |
| Ridge | `penalty="l2"` |
| Lasso | `penalty="l1"` |
| Feature scaling | `StandardScaler()` inside the pipeline |

## The one pattern to memorize

**IMPORT → LOAD → SPLIT → PIPELINE → FIT → PREDICT → METRICS → CROSS-VALIDATION → PLOT**

Start with this skeleton for every practical. Then add only the special code explicitly requested in the question.