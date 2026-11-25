# AML Practicals — Last-Minute Revision

Quick guide to the notebooks and core ideas in this folder. Use the linked notebooks for the full code and outputs.

## Practical index

| Folder | Topic | Last-minute reminders |
|---|---|---|
| [lab01](lab01/) | NumPy | Array shape, dimensions, indexing, creation, and reshape operations. |
| [lab02](lab02/) | Pandas and preprocessing | Inspect data with `head()`, `info()`, and `describe()`; handle missing values and encode categories before modeling. |
| [lab04](lab04/) | Decision tree classification | Diabetes data; `criterion="entropy"` uses information gain. Trees generally do not need feature scaling. |
| [lab05](lab05/) | Linear and regularized regression | Income/happiness data; evaluate with MAE, MSE, and R². `RidgeCV` uses L2 and `LassoCV` uses L1 regularization. |
| [lab06](lab06/) | Logistic regression | Classification; scale features. L1 can drive coefficients to zero, while L2 shrinks them. |
| [lab07](lab07/) | K-nearest neighbors | Scale features because distance drives predictions; the notebook uses `n_neighbors=3`. |
| [lab08](lab08/) | Gaussian Naive Bayes | Probabilistic classifier; review accuracy, confusion matrix, classification report, and cross-validation. |
| [lab09](lab09/) | Clustering | K-means, hierarchical clustering, and silhouette analysis; see the clustering reminders below. |
| [lab10](lab10/) | Ensemble learning | Random forest (bagging) and AdaBoost (boosting). |
| [lab12](lab12/) | PCA | Standardize before dimensionality reduction; PCA components capture directions of high variance. |
| [misc](misc/) | Ridge and polynomial regression questions | Compare training and test MSE across polynomial degrees to discuss bias and variance. |

## The standard supervised-learning workflow

**Load → inspect → define `X` and `y` → split → preprocess → fit → predict → evaluate**

```python
X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

- Keep the test set for final evaluation. Do not fit preprocessing on the full dataset.
- Put scaling, encoding, and other learned preprocessing inside a `Pipeline` so cross-validation does not leak information.
- Scale for distance- or coefficient-sensitive methods such as KNN, logistic regression, and PCA. Decision trees and random forests generally do not require scaling.
- For classification, check accuracy, precision, recall, F1, and the confusion matrix. For regression, check MAE, MSE/RMSE, and R².

## Model and concept reminders

- **Decision tree:** Entropy is `-Σ pᵢ log₂(pᵢ)`; information gain is the reduction in entropy after a split. Gini is another split criterion. Limit growth to reduce overfitting.
- **Linear regression:** Predicts a continuous target. Ridge (L2) shrinks coefficients; Lasso (L1) can set some coefficients to zero. Cross-validation can select `alpha`.
- **Logistic regression:** Classification model; its sigmoid maps scores to probabilities. Scale features and use a suitable solver for L1 regularization.
- **KNN:** Predicts from the nearest training examples. Scale first; compare values of `k` rather than assuming one is optimal.
- **Gaussian Naive Bayes:** Uses Bayes' rule with a conditional-independence assumption and Gaussian feature likelihoods.
- **K-means:** Choose `k` (often with an elbow plot), fit, then inspect cluster labels/centers. Results can depend on initialization; set `random_state` for reproducibility.
- **Hierarchical clustering:** A dendrogram helps select cluster count. Ward linkage merges groups to minimize the increase in within-cluster variance.
- **Silhouette score:** Ranges from -1 to 1; higher generally means better-separated, more cohesive clusters.
- **Bagging / random forest:** Fits models on sampled data and aggregates them to reduce variance.
- **Boosting / AdaBoost:** Builds learners sequentially, focusing more on previously misclassified examples.
- **PCA:** Unsupervised feature transformation. Standardize when features have different units; choose components using explained variance, then fit the downstream model.
- **Bias and variance:** A simple model may underfit (high bias); an overly complex model may overfit (high variance). Compare train and test error.

## Model code and metrics

Use the model that matches the target: a category/class means **classification**; a number to estimate means **regression**; no target column means **clustering** or **dimensionality reduction**.

### Classification

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Choose one model:
model = DecisionTreeClassifier(criterion="entropy", random_state=0)
# model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
# L1 logistic regression:
# model = make_pipeline(StandardScaler(),
#                       LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000))
# model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
# model = GaussianNB()
# model = RandomForestClassifier(n_estimators=100, random_state=0)
# model = AdaBoostClassifier(n_estimators=100, random_state=0)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Metrics for decision trees, logistic regression, KNN, Naive Bayes, random forest, and AdaBoost:**

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
```

| Metric | What it tells you |
|---|---|
| Accuracy | Fraction of predictions that are correct |
| Precision | Of predicted positives, how many were correct |
| Recall | Of actual positives, how many were found |
| F1 | Balance between precision and recall |
| Confusion matrix | Counts correct and incorrect predictions by class |

For imbalanced or multiclass data, consider `average="weighted"` for precision, recall, and F1. For cross-validation, use `cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")`; report the mean score.

### Regression

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

alphas = [0.01, 0.1, 1, 10, 100]
model = LinearRegression()
# model = RidgeCV(alphas=alphas)
# model = LassoCV(alphas=alphas, max_iter=5000)
# Polynomial regression example:
# model = make_pipeline(PolynomialFeatures(degree=2),
#                       StandardScaler(), LinearRegression())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Metrics for linear, Ridge, Lasso, and polynomial regression:**

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("R2:", r2_score(y_test, y_pred))
```

| Metric | What it tells you | Better direction |
|---|---|---|
| MAE | Average absolute prediction error | Lower |
| MSE | Average squared prediction error; penalizes large errors | Lower |
| RMSE | Error in the target's units | Lower |
| R² | Variation explained relative to predicting the mean | Usually higher |

For the polynomial-regression exercise, compare **training and testing MSE** for each degree. A much lower training error than testing error suggests overfitting.

### Clustering and PCA

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
labels = kmeans.fit_predict(X_scaled)
print("K-means inertia:", kmeans.inertia_)
print("Silhouette:", silhouette_score(X_scaled, labels))

# Hierarchical alternative; use a dendrogram to help choose n_clusters.
hierarchical = AgglomerativeClustering(n_clusters=5, linkage="ward")
hierarchical_labels = hierarchical.fit_predict(X_scaled)
print("Hierarchical silhouette:", silhouette_score(X_scaled, hierarchical_labels))

# PCA: inspect how much variance the selected components retain.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", pca.explained_variance_ratio_.sum())
```

| Method | Useful evaluation |
|---|---|
| K-means | Inertia for the elbow plot; silhouette score for cluster cohesion/separation |
| Hierarchical clustering | Dendrogram to choose cluster count; silhouette score after assigning clusters |
| PCA | Explained variance ratio; if followed by a classifier, evaluate that classifier with classification metrics |

Clustering has no `y_test` labels in these exercises, so do not use accuracy, precision, recall, or F1 unless known ground-truth labels are available.

## Data files

CSV datasets are in [data](data/):

| File | Quick description |
|---|---|
| `Data.csv` | Country, age, salary, and purchase outcome |
| `diabetes.csv` | Medical features and diabetes outcome |
| `income_data.csv` | Income and happiness |
| `mall.csv` | Customer demographics, income, and spending score |
| `mushrooms.csv` | Mushroom characteristics and class |
| `Social_Network_Ads.csv` | Age, estimated salary, and purchase outcome |

Some notebooks contain machine-specific absolute file paths. If a notebook cannot find a CSV, run it from this folder and use a relative path, for example:

```python
from pathlib import Path

data_path = Path("data") / "diabetes.csv"
data = pd.read_csv(data_path)
```

## Before running a notebook

Use a Python environment with Jupyter, NumPy, pandas, Matplotlib, and scikit-learn installed. Run notebook cells from top to bottom, and verify that the target column is excluded from `X` before fitting a model.
