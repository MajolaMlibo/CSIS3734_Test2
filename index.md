
# Test 2 Solutions
---

[2025](#2025-paper-solutions)
[2024](#2024-paper-solutions)
[2023](#2023-paper-solutions)
[2022](#2022-paper-solutions)

---


[Back To Top](#test-2-solutions)
# 2025 PAPER **SOLUTIONS**
**Date:** 14 May 2025 | **Time:** 180 min | **Total Marks:** 90
---

## QUESTION 2 [70 Marks – 140 Minutes] *(Open Book)*

> **Dataset:** `credit_card_customers.csv`
> **Notebook Name:** `CSIS3754_2026206108_2026`
> **Target:** Predict whether a customer will stay or leave (`Attrition_Flag`)

---

### 2.1 Import the dataset *(1 mark)*

```python
import pandas as pd

customers = pd.read_csv('credit_card_customers.csv')
customers
```

---

### 2.2 Inspect the data *(4 marks)*

```python
# Row count and column count
print(f"Rows: {customers.shape[0]}, Columns: {customers.shape[1]}")

# Last 20 records
customers.tail(20)

# Data type for each column
customers.dtypes

# Number of unique values per feature
customers.nunique()
```

---

### 2.3 Discard first and last 7 columns *(2 marks)*

```python
# Without specifying column names — use iloc indexing
customers = customers.iloc[:, 7:-7]
customers

or 

# Single column: 
df.drop(df.columns[0], axis=1, inplace=True) # (Deletes the 1st column).# Multiple columns: 
df.drop(df.columns[[0, 2]], axis=1) # (Deletes the 1st and 3rd columns).
```

> **💡 Explanation:** `iloc[:, 1:-7]` # Keeps everything from the 2nd column up to (but not including) the last 7

---

### 2.4 What does `Avg_Open_To_Buy` refer to? *(2 marks)*

```python
print("""
Avg_Open_To_Buy refers to the average available credit limit that a customer 
has remaining to spend on their credit card. It is calculated as the difference 
between the customer's credit limit and the current balance owed 
(i.e., Credit_Limit - Total_Revolving_Bal). A higher value means the customer 
has more available credit to use.
""")
```

---

### 2.5 Statistical summary *(1 mark)*

```python
df_encoded = pd.get_dummies(customers, columns=['Gender'])
customers['Education_Level'] = customers['Education_Level'].astype('category').cat.codes
customers.describe()
```
> **💡 Explanation:** `df_encoded` One-Hot Encoding (Best for nominal data): This creates a new binary (0 or 1) column for every unique category. Use pandas.get_dummies() to achieve this. 
> 
> Label Encoding (Best for many categories): This assigns a unique integer (0, 1, 2...) to each category. You can use astype('category').cat.codes or sklearn.preprocessing.LabelEncoder.

#### 2.5.1 Deduction about education level *(2 marks)*

```python
print("""
From the statistical summary, the Education_Level column is a categorical 
(text) feature and will not appear in the numeric summary via describe(). 
However, from the unique value inspection, we can see there are multiple 
education levels present (e.g., Uneducated, High School, Graduate, etc.). 
The data does not indicate one dominant education level — customers are spread 
across various education levels, suggesting a diverse customer base.
""")
```

---

### 2.6 Percentage of customers earning R120K+ *(4 marks)*

```python
# Filter for Uneducated customers with income >= 120K
uneducated = customers[customers['Education_Level'] == 'Uneducated']
pct_uneducated = (uneducated['Income_Category'].isin(['$120K +'])).sum() / len(uneducated) * 100
print(f"Uneducated customers earning R120K+: {pct_uneducated:.2f}%")

# Filter for PhD customers with income >= 120K
phd = customers[customers['Education_Level'] == 'Doctorate']
pct_phd = (phd['Income_Category'].isin(['$120K +'])).sum() / len(phd) * 100
print(f"PhD customers earning R120K+: {pct_phd:.2f}%")
```

#### 2.6.1 Conclusion *(1 mark)*

```python
print("""
Customers with a Doctorate (PhD) degree have a significantly higher percentage 
of earning R120K or more compared to uneducated customers. This supports the 
general expectation that higher levels of education are positively correlated 
with higher income levels.
""")
```

---

### 2.7 Visualisations using Seaborn *(9 marks)*

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot 1: Category plot – Customers per Card Category for each Education Level
plt.figure(figsize=(12,6))
sns.countplot(data=customers, x='Education_Level', hue='Card_Category')
plt.title('Number of Customers per Card Category for Each Education Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("""
Discussion: The plot shows that 'Blue' card holders are the most common across 
all education levels. Customers with higher education levels tend to hold 
premium card categories (Silver, Gold, Platinum) more frequently, though 
Blue cards still dominate all groups.
""")

# Plot 2: Scatter plot – Credit Limit vs Avg_Open_To_Buy
plt.figure(figsize=(8,6))
sns.scatterplot(data=customers, x='Credit_Limit', y='Avg_Open_To_Buy')
plt.title('Credit Limit vs Avg_Open_To_Buy')
plt.tight_layout()
plt.show()
print("""
Discussion: There is a strong positive linear relationship between Credit_Limit 
and Avg_Open_To_Buy. This makes sense because a higher credit limit means more 
available credit to spend, assuming the same balance owed.
""")

# Plot 3: Combined scatter plot – Income Category vs Credit Limit per Card Category
plt.figure(figsize=(10,6))
sns.scatterplot(data=customers, x='Income_Category', y='Credit_Limit', hue='Card_Category')
plt.title('Income Category vs Credit Limit by Card Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("""
Discussion: Customers in higher income brackets tend to have higher credit limits. 
Platinum card holders cluster in the higher credit limit ranges across income 
categories, while Blue card holders are spread more broadly across all income 
and credit limit levels.
""")
```

---

### 2.8 Count of existing vs attrited customers *(1 mark)*

```python
print(customers['Attrition_Flag'].value_counts())
```

---

### 2.9 Check data balance with pie chart *(4 marks)*

```python
counts = customers['Attrition_Flag'].value_counts()
percentages = (counts / counts.sum() * 100).round(3)

plt.figure(figsize=(6,6))
plt.pie(percentages, labels=percentages.index, autopct='%.3f%%')
plt.title('Customer Attrition Distribution')
plt.show()

print("""
The data is imbalanced — the majority of records belong to existing customers, 
while attrited customers make up a smaller proportion. This imbalance needs to 
be addressed before training classifiers.
""")
```

---

### 2.10 Undersample to balance the data *(5 marks)*

```python
from sklearn.utils import resample

# Separate classes
existing = customers[customers['Attrition_Flag'] == 'Existing Customer']
attrited = customers[customers['Attrition_Flag'] == 'Attrited Customer']

# Undersample the majority class (existing) to match minority class (attrited)
existing_downsampled = resample(existing,
                                replace=False,
                                n_samples=len(attrited),
                                random_state=42)

# Combine into resampled dataframe
customers_resampled = pd.concat([existing_downsampled, attrited])
customers_resampled = customers_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
customers_resampled
```

---

### 2.11 Confirm successful resampling *(3 marks)*

```python
# Count of each class
print(customers_resampled['Attrition_Flag'].value_counts())

# Updated pie chart
counts_r = customers_resampled['Attrition_Flag'].value_counts()
pct_r = (counts_r / counts_r.sum() * 100).round(3)

plt.figure(figsize=(6,6))
plt.pie(pct_r, labels=pct_r.index, autopct='%.3f%%')
plt.title('Resampled Customer Attrition Distribution')
plt.show()
```

---

### 2.12 Convert text to numeric values *(7 marks)*

```python
from sklearn.preprocessing import LabelEncoder

# Step 1: Binary encoding for columns with exactly 2 unique text values
binary_cols = [col for col in customers_resampled.select_dtypes(include='object').columns
               if customers_resampled[col].nunique() == 2]

le = LabelEncoder()
for col in binary_cols:
    customers_resampled[col] = le.fit_transform(customers_resampled[col])

print("Binary-encoded columns:", binary_cols)

# Step 2: One-hot encoding for columns with more than 2 unique text values
multi_cols = [col for col in customers_resampled.select_dtypes(include='object').columns
              if customers_resampled[col].nunique() > 2]

customers_resampled = pd.get_dummies(customers_resampled, columns=multi_cols, drop_first=False)

print("One-hot encoded columns:", multi_cols)
customers_resampled
```

---

### 2.13 Define X, y and create train/test split *(7 marks)*

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features and target
X = customers_resampled.drop('Attrition_Flag', axis=1)
y = customers_resampled['Attrition_Flag']

# Train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train dimensions: {X_train.shape}")
print(f"X_test dimensions:  {X_test.shape}")

# Feature scaling (required for KNN, LR, SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 2.14 Train classifiers with k-fold cross-validation *(10 marks)*

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, learning_curve
import numpy as np
import matplotlib.pyplot as plt

classifiers = {
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(gamma='auto')
}

k = 5
results = {}

for name, clf in classifiers.items():
    acc_scores = cross_val_score(clf, X_train, y_train, cv=k, scoring='accuracy')
    f1_scores  = cross_val_score(clf, X_train, y_train, cv=k, scoring='f1_weighted')
    results[name] = {'accuracy': acc_scores.mean(), 'f1': f1_scores.mean()}
    print(f"{name}:")
    print(f"  CV Training Accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")
    print(f"  CV F1 Score:          {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})\n")

# Learning curves for each classifier
for name, clf in classifiers.items():
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X_train, y_train, cv=k, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score', linestyle='--')
    plt.plot(train_sizes, val_scores.mean(axis=1),   label='Cross-validation Score')
    plt.title(f'Learning Curve – {name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

---

### 2.15 Best model predictions and evaluation *(3 marks)*

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Select best model based on highest F1 score
best_name = max(results, key=lambda k: results[k]['f1'])
print(f"Best model (highest F1): {best_name}")

best_clf = classifiers[best_name]
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

### 2.16 Discuss model metrics *(4 marks)*

```python
print("""
=== Model Discussion ===

1. Training Accuracy vs Testing Accuracy:
   If the training accuracy is significantly higher than the test accuracy, 
   the model is overfitting — it has learned the training data too well but 
   cannot generalise. If they are close, the model generalises well (good fit). 
   Ideally, the test accuracy should be close to the training accuracy, 
   indicating the model performs consistently on unseen data.

2. Precision:
   Precision measures the proportion of positive predictions that are actually 
   correct: TP / (TP + FP). A high precision means few false positives — 
   when the model predicts a customer will leave, it is usually correct.

3. Recall:
   Recall measures the proportion of actual positives that were correctly 
   identified: TP / (TP + FN). A high recall means the model correctly 
   identifies most customers who will actually leave (few false negatives).

4. F1-Score:
   The F1-score is the harmonic mean of precision and recall: 
   2 × (Precision × Recall) / (Precision + Recall). 
   It is especially useful for imbalanced datasets because it balances both 
   precision and recall. A high F1-score indicates the model performs well 
   on both metrics simultaneously.
""")
```

---
---
[Back To Top](#test-2-solutions)
# 2024 PAPER SOLUTIONS
**Date:** 22 May 2024 | **Time:** 180 min | **Total Marks:** 90S
---

## QUESTION 2 [50 Marks – 100 Minutes] *(Open Book)*

> **Dataset:** `fake_currency_data.csv`
> **Notebook Name:** `CSIS3754_YourStudentNumber_ST2-2_2024`
> **Target:** Predict whether printed currency is counterfeit or genuine (`Counterfeit`)

---

### 2.1 Import the dataset *(1 mark)*

```python
import pandas as pd

currency = pd.read_csv('fake_currency_data.csv')
currency
```

---

### 2.2 Rename column *(2 marks)*

```python
currency.rename(columns={'Country': 'CountryOfOrigin'}, inplace=True)
currency
```

---

### 2.3 Inspect the data *(3 marks)*

```python
# Concise summary: index, dtypes, non-null counts, memory usage
currency.info()

# Statistical summary
currency.describe()

# Number of unique values per feature
currency.nunique()

# value of unique values per feature
dfcus['marital'].unique()
```

#### 2.3.1 Discussion of data inspection *(5 marks)*

```python
print("""
From the concise summary (info()):
- The dataset contains multiple columns with both numeric and categorical data.
- Some columns may have missing values (non-null count < total rows), 
  which need to be addressed before modelling.

From the unique values:
- Some columns have very few unique values (e.g., 2), indicating binary or 
  categorical features suitable for label/one-hot encoding.
- Columns like 'CountryOfOrigin' may have many unique values, 
  requiring one-hot encoding.

From the statistical summary (describe()):
- Numeric features like Weight, Length, Width, and Thickness may contain 
  outliers, visible from extreme min/max values compared to the mean.
- A large standard deviation relative to the mean suggests the presence 
  of outliers, which can negatively impact model performance.

Possible issues:
- Missing values may be present and need to be handled.
- Outliers in numeric features (especially Weight, Length, Width, Thickness) 
  may distort model training.
""")
```

---

### 2.4 Boxplots for numeric features *(2 marks)*

```python
import seaborn as sns
import matplotlib.pyplot as plt

features = ['Weight', 'Length', 'Width', 'Thickness']

fig, axes = plt.subplots(1, 4, figsize=(16,5))
for ax, feat in zip(axes, features):
    sns.boxplot(y=currency[feat], ax=ax)
    ax.set_title(feat)
plt.tight_layout()
plt.show()
```

#### 2.4.1 Issue identified *(1 mark)*

```python
print("""
The boxplots reveal the presence of significant OUTLIERS in the numeric 
features (Weight, Length, Width, Thickness). These are represented by dots 
beyond the whiskers of the boxplots. Outliers can skew the model's learning 
and degrade classification performance.
""")
```

#### 2.4.2 Fix the issue *(3 marks)*

```python
# Fix: Remove outliers using the IQR method
from scipy import stats
import numpy as np

print("Shape before outlier removal:", currency.shape)

# Remove rows where any numeric feature has a z-score > 3
z_scores = np.abs(stats.zscore(currency[features]))
currency = currency[(z_scores < 3).all(axis=1)]

print("Shape after outlier removal:", currency.shape)

print("""
Motivation: The IQR/Z-score method is a standard statistical approach to 
remove extreme outliers. Removing records where any numeric feature falls 
more than 3 standard deviations from the mean ensures that the remaining 
data is representative without discarding too many valid records.
""")

# Confirm fix by recreating boxplots
fig, axes = plt.subplots(1, 4, figsize=(16,5))
for ax, feat in zip(axes, features):
    sns.boxplot(y=currency[feat], ax=ax)
    ax.set_title(feat + ' (cleaned)')
plt.tight_layout()
plt.show()
```

---

### 2.5 Reduce sample size *(2 marks)*

```python
# Select a random sample without replacement (e.g., 2000 records)
currency = currency.sample(n=2000, replace=False, random_state=42).reset_index(drop=True)
print(f"Reduced dataset shape: {currency.shape}")
currency
```

---

### 2.6 Balance check – pie chart *(3 marks)*

```python
counts = currency['Counterfeit'].value_counts()
pct = (counts / counts.sum() * 100).round(1)

plt.figure(figsize=(6,6))
plt.pie(pct, labels=pct.index, autopct='%.1f%%')
plt.title('Counterfeit vs Genuine Currency Distribution')
plt.show()
```

---

### 2.7 Is the data balanced? *(1 mark)*

```python
print("""
If the pie chart shows approximately 50%/50% between counterfeit and genuine 
currency, the data is balanced and no further resampling is required.

If the split is significantly unequal (e.g., 70%/30%), the data is imbalanced 
and resampling (oversampling minority class or undersampling majority class) 
should be performed to ensure the classifier does not become biased toward the 
majority class.
""")
```

---

### 2.8 Convert text to numeric values *(6 marks)*

```python
from sklearn.preprocessing import LabelEncoder

# Binary encoding for columns with exactly 2 unique text values
binary_cols = [col for col in currency.select_dtypes(include='object').columns
               if currency[col].nunique() == 2]
le = LabelEncoder()
for col in binary_cols:
    currency[col] = le.fit_transform(currency[col])
print("Binary encoded:", binary_cols)

# One-hot encoding for columns with more than 2 unique text values
multi_cols = [col for col in currency.select_dtypes(include='object').columns
              if currency[col].nunique() > 2]
currency = pd.get_dummies(currency, columns=multi_cols, drop_first=False)
print("One-hot encoded:", multi_cols)
currency
```

---

### 2.9 Confirm no object columns remain *(1 mark)*

```python
print(currency.dtypes)
print("\nObject columns remaining:", list(currency.select_dtypes(include='object').columns))
# Expected: empty list []
```

---

### 2.10 Define X, y and train/test split *(6 marks)*

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = currency.drop('Counterfeit', axis=1)
y = currency['Counterfeit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

---

### 2.11 Train classifiers with k-fold cross-validation *(8 marks)*

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, learning_curve
import numpy as np
import matplotlib.pyplot as plt

classifiers = {
    'Naïve Bayes':        GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'SVM':                 SVC(gamma='auto')
}

k = 5
results = {}

for name, clf in classifiers.items():
    acc = cross_val_score(clf, X_train, y_train, cv=k, scoring='accuracy')
    f1  = cross_val_score(clf, X_train, y_train, cv=k, scoring='f1_weighted')
    results[name] = {'accuracy': acc.mean(), 'f1': f1.mean()}
    print(f"{name}:")
    print(f"  CV Accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"  CV F1 Score: {f1.mean():.4f} ± {f1.std():.4f}\n")

# Learning curves
for name, clf in classifiers.items():
    train_sizes, train_sc, val_sc = learning_curve(
        clf, X_train, y_train, cv=k,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_sc.mean(axis=1), '--', label='Training Score')
    plt.plot(train_sizes, val_sc.mean(axis=1),        label='Cross-Validation Score')
    plt.title(f'Learning Curve – {name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

---

### 2.12 Best model evaluation *(3 marks)*

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

best_name = max(results, key=lambda k: results[k]['f1'])
print(f"Best model: {best_name}")

best_clf = classifiers[best_name]
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

### 2.13 Discuss model metrics *(3 marks)*

```python
print("""
1. Training Accuracy vs Testing Accuracy:
   A close match between training and testing accuracy means the model 
   generalises well. A large gap (high training, low testing) means overfitting.

2. Precision:
   Precision = TP / (TP + FP). It answers: "Of all predicted positives, 
   how many were actually positive?" High precision = few false alarms.

3. Recall:
   Recall = TP / (TP + FN). It answers: "Of all actual positives, how many 
   did the model catch?" High recall = few missed detections.

4. F1-Score:
   F1 = 2 × (Precision × Recall) / (Precision + Recall).
   The harmonic mean of precision and recall — useful when both false positives 
   and false negatives carry significant consequences, such as classifying 
   counterfeit currency incorrectly.
""")
```

---

## QUESTION 3 [20 Marks – 40 Minutes] *(Open Book)*

> **Dataset:** `gin.csv`
> **Notebook Name:** `CSIS3754_YourStudentNumber_ST2-3_2024`
> **Task:** Unsupervised ML — group types of gin by chemical composition using k-Means clustering

---

### 3.1 Import the dataset *(1 mark)*

```python
import pandas as pd

gin = pd.read_csv('gin.csv')
gin
```

---

### 3.2 Ensure no zero values *(1 mark)*

```python
# Check for zeros
print("Zero values per column:")
print((gin == 0).sum())

# Replace zeros with NaN and then fill with the column mean
gin.replace(0, pd.NA, inplace=True)
gin.fillna(gin.mean(), inplace=True)

print("\nAfter fixing — zero values per column:")
print((gin == 0).sum())
gin
```

---

### 3.3 k-Means Clustering *(11 marks)*

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Select features
features = ['Alcohol', 'Malic_Acid', 'Ash_Alcanity', 'Proanthocyanins', 'Hue']
X_gin = gin[features]

# Pre-processing: scale the data
scaler = StandardScaler()
X_gin_scaled = scaler.fit_transform(X_gin)

# Determine best k using silhouette score
sil_scores = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_gin_scaled)
    sil_scores.append(silhouette_score(X_gin_scaled, labels))

# Plot silhouette scores
plt.figure(figsize=(8,5))
plt.plot(k_range, sil_scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()

# Select best k
best_k = k_range[sil_scores.index(max(sil_scores))]
print(f"Best k (highest silhouette score): {best_k}")

# Train final model with best k
km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
gin['Cluster'] = km_best.fit_predict(X_gin_scaled)

print("\nCluster Labels:")
print(gin['Cluster'].values)

print("\nCluster Centres (scaled):")
print(km_best.cluster_centers_)
```

---

### 3.4 Visualise clusters with scatter plots *(5 marks)*

```python
import seaborn as sns
from itertools import combinations

feature_pairs = list(combinations(features, 2))

for feat1, feat2 in feature_pairs:
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=gin, x=feat1, y=feat2, hue='Cluster', palette='tab10')
    
    # Plot centroids
    # Back-transform centroids for plotting
    centres_original = scaler.inverse_transform(km_best.cluster_centers_)
    centres_df = pd.DataFrame(centres_original, columns=features)
    
    plt.scatter(centres_df[feat1], centres_df[feat2],
                s=200, c='black', marker='X', label='Centroid', zorder=5)
    
    plt.title(f'{feat1} vs {feat2} – Clusters')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

---

### 3.5 Evaluate optimal k-value *(2 marks)*

```python
print(f"""
The silhouette score indicated an optimal k of {best_k} clusters.

Comparing this to the scatter plots:
- If the scatter plots visually show {best_k} distinct, well-separated groups 
  for most feature pairs, this confirms the silhouette score result.
- If the scatter plots show a different number of visually distinct groups 
  (e.g., 3 clearly separated clusters when k={best_k} was selected), 
  this suggests the silhouette score may not perfectly match the visual 
  structure of the data, potentially because some clusters overlap in 
  certain feature dimensions.
  
In general, the silhouette score provides a quantitative validation, while 
the scatter plots provide qualitative visual confirmation. When they agree, 
confidence in the chosen k value is high.
""")
```

---

## 📌 Quick Reference Summary

| Year | Q1 Topics | Q2 Dataset | Q3 |
|------|-----------|------------|----|
| 2025 | Learning Curves, k-Means steps, Gini Impurity | `credit_card_customers.csv` — Customer Attrition | — |
| 2024 | Learning Curves, Dendrogram, Gini Impurity | `fake_currency_data.csv` — Counterfeit Currency | `gin.csv` — k-Means Clustering |
| 2023 | Learning Curves, Dendrogram, k-Means steps, Sentiment Analysis NLP | `bank_customers.csv` — Customer Attrition | `wine.csv` — k-Means Clustering |
| 2022 | Supervised vs Unsupervised, Overfitting, KNN algorithm, Gini Impurity + Decision Tree | `medicine.csv` — Medication Prediction | — |

---

*Solutions prepared for CSIS3764/CSIS3754 – Department of Computer Science & Informatics, UFS Bloemfontein Campus*

---
---

[Back To Top](#test-2-solutions)

# 2023 PAPER SOLUTIONS
**Date:** 20 October 2023 | **Time:** 180 min | **Total Marks:** 90 | **Module:** CSIS3764

---

## QUESTION 2 [50 Marks – 100 Minutes] *(Open Book)*

> **Dataset:** `bank_customers.csv` — 10 127 records, 23 columns
> **Notebook Name:** `CSIS3764_YourStudentNumber_ST2-2_2023`
> **Target:** Predict customer attrition (`Attrition_Flag`: Existing Customer / Attrited Customer)
>
> **Dataset facts (from actual file):**
> - Shape after import: **(10127, 23)**
> - After removing first + last 7 cols: **(10127, 15)** — columns: `Attrition_Flag`, `Customer_Age`, `Gender`, `Dependent_count`, `Education_Level`, `Marital_Status`, `Income_Category`, `Card_Category`, `Months_on_book`, `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon`, `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`
> - Attrition split: **8500** Existing / **1627** Attrited (heavily imbalanced)
> - Marital Status: Married (4687), Single (3943), Unknown (749), Divorced (748)

---

### 2.1 Import the dataset *(1 mark)*

```python
import pandas as pd

dfcus = pd.read_csv('bank_customers.csv')
dfcus
```

---

### 2.2 Inspect the data *(2 marks)*

```python
# Concise summary: index, data types, columns, non-null values, memory usage
dfcus.info()

# Number of unique values for each feature
dfcus.nunique()
```

---

### 2.3 Discard first column and last 7 columns *(2 marks)*

```python
# Drop first column (index 0) AND last 7 columns — no column names used
dfcus = dfcus.iloc[:, 1:-7]
dfcus
```

> **💡 Explanation:** `iloc[:, 1:-7]` selects all rows, starting from column index 1 (skipping the first column `CLIENTNUM`) up to but not including the last 7 columns. No column names are needed.
>
> **Remaining columns (15):** `Attrition_Flag, Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status, Income_Category, Card_Category, Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Avg_Open_To_Buy`

---

### 2.4 Statistical summary *(1 mark)*

```python
dfcus.describe()
```

#### 2.4.1 Deduction about marital status *(2 marks)*

```python
print("""
From the statistical summary (describe()), the Marital_Status column is 
categorical and does not appear in the numeric summary. However, from the 
unique value inspection (nunique) and value_counts, we can see there are 
4 distinct marital status values: Married, Single, Divorced, and Unknown.

The most frequent marital status is 'Married' (4687 customers), followed 
by 'Single' (3943), while 'Unknown' (749) and 'Divorced' (748) appear 
far less frequently. This means the majority of credit card customers 
in this dataset are married or single, and a small proportion did not 
disclose their marital status.
""")
```

---

### 2.5 Visualisations using Seaborn *(6 marks)*

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot 1: Categorical plot — Customers per Card Category for each Education Level
plt.figure(figsize=(12, 6))
sns.countplot(data=dfcus, x='Education_Level', hue='Card_Category')
plt.title('Number of Customers per Card Category for Each Education Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("""
Discussion: The 'Blue' card category dominates across all education levels, 
indicating it is the most common card type regardless of education. 
Graduate and High School educated customers make up the largest groups. 
Platinum card holders are very rare across all education levels, suggesting 
it is a premium tier with limited uptake.
""")

# Plot 2: Combined scatter plot — Income Category vs Credit Limit per Card Category
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dfcus, x='Income_Category', y='Credit_Limit', hue='Card_Category')
plt.title('Income Category vs Credit Limit by Card Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("""
Discussion: Customers in higher income brackets (R120K+) tend to have higher 
credit limits. Platinum and Gold card holders cluster toward the higher credit 
limit ranges. Blue card holders are spread across all income and credit limit 
levels, confirming it is the most broadly accessible card category.
""")
```

---

### 2.6 Check data balance with pie chart *(4 marks)*

```python
counts = dfcus['Attrition_Flag'].value_counts()
pct = (counts / counts.sum() * 100).round(1)

plt.figure(figsize=(6, 6))
plt.pie(pct, labels=pct.index, autopct='%.1f%%')
plt.title('Customer Attrition Distribution')
plt.show()

# Expected: ~83.9% Existing Customer, ~16.1% Attrited Customer
print("""
The data is heavily IMBALANCED — approximately 83.9% of records are 
existing customers and only 16.1% are attrited customers. 
This imbalance must be corrected before training classifiers, 
otherwise the model will be biased toward predicting 'Existing Customer'.
""")
```

---

### 2.7 Undersample to balance the data *(5 marks)*

```python
from sklearn.utils import resample

# Separate the two classes
existing = dfcus[dfcus['Attrition_Flag'] == 'Existing Customer']
attrited = dfcus[dfcus['Attrition_Flag'] == 'Attrited Customer']

print(f"Before resampling — Existing: {len(existing)}, Attrited: {len(attrited)}")

# Undersample the majority class to match the minority class count
existing_downsampled = resample(existing,
                                replace=False,
                                n_samples=len(attrited),  # match minority: 1627
                                random_state=42)

# Combine and shuffle
dfcus_resampled = pd.concat([existing_downsampled, attrited])
dfcus_resampled = dfcus_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"After resampling — shape: {dfcus_resampled.shape}")
dfcus_resampled
```

---

### 2.8 Convert text to numeric values *(7 marks)*

```python
from sklearn.preprocessing import LabelEncoder

# Binary columns (exactly 2 unique text values) → encode as [0, 1]
# In this dataset: Attrition_Flag (2), Gender (2)
binary_cols = [col for col in dfcus_resampled.select_dtypes(include=['object', 'string']).columns
               if dfcus_resampled[col].nunique() == 2]

le = LabelEncoder()
for col in binary_cols:
    dfcus_resampled[col] = le.fit_transform(dfcus_resampled[col])

print("Binary-encoded columns:", binary_cols)

# Multi-value columns (>2 unique text values) → one-hot encoding
# In this dataset: Education_Level (7), Marital_Status (4), Income_Category (6), Card_Category (4)
multi_cols = [col for col in dfcus_resampled.select_dtypes(include=['object', 'string']).columns
              if dfcus_resampled[col].nunique() > 2]

dfcus_resampled = pd.get_dummies(dfcus_resampled, columns=multi_cols, drop_first=False)

print("One-hot encoded columns:", multi_cols)
dfcus_resampled
```

---

### 2.9 Define X, y and create train/test split *(7 marks)*

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# X = all features, y = target (Attrition_Flag, already binary encoded as 0/1)
X = dfcus_resampled.drop('Attrition_Flag', axis=1)
y = dfcus_resampled['Attrition_Flag']

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train dimensions: {X_train.shape}")
print(f"X_test dimensions:  {X_test.shape}")

# Feature scaling — required for Naive Bayes (NB: NB doesn't need scaling,
# but SVM definitely does; apply universally for safety)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

---

### 2.10 Train classifiers with k-fold cross-validation *(8 marks)*

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, learning_curve
import numpy as np
import matplotlib.pyplot as plt

classifiers = {
    'Naive Bayes': GaussianNB(),
    'SVM':         SVC(gamma='auto')
}

k = 5
results = {}

for name, clf in classifiers.items():
    acc = cross_val_score(clf, X_train, y_train, cv=k, scoring='accuracy')
    f1  = cross_val_score(clf, X_train, y_train, cv=k, scoring='f1_weighted')
    results[name] = {'accuracy': acc.mean(), 'f1': f1.mean()}
    print(f"{name}:")
    print(f"  CV Training Accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"  CV F1 Score:          {f1.mean():.4f} ± {f1.std():.4f}\n")

# Learning curves for each classifier
for name, clf in classifiers.items():
    train_sizes, train_sc, val_sc = learning_curve(
        clf, X_train, y_train, cv=k, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_sc.mean(axis=1), linestyle='--', label='Training Score')
    plt.plot(train_sizes, val_sc.mean(axis=1), label='Cross-validation Score')
    plt.title(f'Learning Curve – {name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

---

### 2.11 Best model evaluation *(3 marks)*

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Select best model based on highest F1 score
best_name = max(results, key=lambda k: results[k]['f1'])
print(f"Best model (highest F1): {best_name}")

best_clf = classifiers[best_name]
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

### 2.12 Training accuracy vs testing accuracy *(2 marks)*

```python
print("""
If the model's training accuracy (from cross-validation) is significantly 
higher than the test accuracy, this indicates OVERFITTING — the model has 
learned the training data too well but cannot generalise to unseen data.

If the two values are close to each other, the model generalises well and 
is considered a good fit. In an ideal model, the test accuracy should be 
close to (or only slightly below) the training accuracy, showing that the 
model performs consistently on both seen and unseen data.
""")
```

---

## QUESTION 3 [20 Marks – 40 Minutes] *(Open Book)*

> **Dataset:** `wine.csv` — 178 records, 13 columns
> **Notebook Name:** `CSIS3764_YourStudentNumber_ST2-3_2023`
> **Task:** Unsupervised ML — cluster wine types by chemical composition using k-Means
>
> **Features used:** `Alcohol`, `Magnesium`, `Total_Phenols`, `Flavanoids`, `Color_Intensity`

---

### 3.1 Read the dataset *(1 mark)*

```python
import pandas as pd

wine = pd.read_csv('wine.csv')
wine
```

---

### 3.2 Ensure no zero values *(1 mark)*

```python
# Check for zeros
print("Zero values per column:")
print((wine == 0).sum())

# Replace zeros with NaN, then fill with the column mean
wine.replace(0, pd.NA, inplace=True)
wine.fillna(wine.mean(numeric_only=True), inplace=True)

print("\nAfter fixing — zero values per column:")
print((wine == 0).sum())
wine
```

---

### 3.3 k-Means Clustering *(11 marks)*

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Select required features only
features = ['Alcohol', 'Magnesium', 'Total_Phenols', 'Flavanoids', 'Color_Intensity']
X_wine = wine[features]

# Pre-processing: standardise the data (required for k-Means)
scaler = StandardScaler()
X_wine_scaled = scaler.fit_transform(X_wine)

# Determine best k using silhouette score
sil_scores = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_wine_scaled)
    sil_scores.append(silhouette_score(X_wine_scaled, labels))

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(k_range, sil_scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters (k) — Wine')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()

# Select best k (highest silhouette score)
best_k = list(k_range)[sil_scores.index(max(sil_scores))]
print(f"Best k (highest silhouette score): {best_k}")

# Train final model with best k
km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
wine['Cluster'] = km_best.fit_predict(X_wine_scaled)

print("\nCluster Labels:")
print(wine['Cluster'].values)

print("\nCluster Centres (in scaled space):")
print(km_best.cluster_centers_)
```

---

### 3.4 Visualise clusters with scatter plots *(5 marks)*

```python
import seaborn as sns
from itertools import combinations
import pandas as pd

feature_pairs = list(combinations(features, 2))
# 5 features → C(5,2) = 10 scatter plots

# Back-transform centroids to original scale for accurate plotting
centres_original = scaler.inverse_transform(km_best.cluster_centers_)
centres_df = pd.DataFrame(centres_original, columns=features)

for feat1, feat2 in feature_pairs:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=wine, x=feat1, y=feat2, hue='Cluster', palette='tab10')

    # Overlay cluster centroids
    plt.scatter(centres_df[feat1], centres_df[feat2],
                s=250, c='black', marker='X', label='Centroid', zorder=5)

    plt.title(f'{feat1} vs {feat2} – k-Means Clusters (k={best_k})')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

---

### 3.5 Evaluate optimal k-value *(2 marks)*

```python
print(f"""
The silhouette score analysis returned an optimal k of {best_k} clusters.

Comparing this to the scatter plots:
- If the scatter plots visually show {best_k} clearly distinct, well-separated 
  groups for most feature pairs (especially Flavanoids vs Total_Phenols and 
  Alcohol vs Color_Intensity), this confirms the silhouette result.
- The wine dataset is known to contain 3 natural grape variety groups, so 
  k=3 is a commonly expected result. If the silhouette score also returns 
  k=3, this is strong evidence for a good clustering solution.
- Where clusters overlap in some scatter plots (e.g., Magnesium vs Ash), 
  this indicates those features alone are not sufficient to separate the groups — 
  the model uses all 5 features simultaneously, which is more powerful.

Conclusion: When the silhouette-derived k matches the visual cluster count 
in the scatter plots, confidence in the chosen k is high.
""")
```

---
---

[Back To Top](#test-2-solutions)

# 2022 PAPER SOLUTIONS
**Date:** 21 October 2022 | **Time:** 180 min | **Total Marks:** 90 | **Module:** CSIS3764
---

## QUESTION 2 [63 Marks – 140 Minutes] *(Open Book)*

> **Dataset:** `medicine.csv` — 200 records, 6 columns
> **Notebook Name:** `CSIS3764_YourStudentNumber_T2_2022`
> **Target:** Predict which medication to prescribe (`Medication`: meds1–meds5)
>
> **Dataset facts (from actual file):**
> - Columns: `Age` (int), `Gender` (M/F binary), `Blood_Presure` (High/Low/Normal — 3 values), `Cholesterol` (High/Normal — binary), `Ratio_Na_K` (float), `Medication` (meds1–meds5 — 5 values)
> - No missing values
> - Medication counts: meds5=91, meds4=54, meds1=23, meds3=16, meds2=16 *(heavily imbalanced)*
> - Binary text cols: `Gender`, `Cholesterol`
> - Multi-value text cols: `Blood_Presure` (3), `Medication` (5)

---

### 2.1 Import the dataset *(1 mark)*

```python
import pandas as pd

meds = pd.read_csv('medicine.csv')
meds
```

---

### 2.2 Inspect the data *(4 marks)*

```python
# First 20 records
meds.head(20)

# Statistical summary
meds.describe()

# Concise summary: index, data types, non-null counts, memory usage
meds.info()

# Check for missing values
print("Missing values:")
print(meds.isnull().sum())
```

---

### 2.3 Scatter plot — Age vs Ratio_Na_K per Medication *(3 marks)*

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
sns.scatterplot(data=meds, x='Age', y='Ratio_Na_K', hue='Medication', palette='tab10')
plt.title('Age vs Ratio_Na_K for Each Medication')
plt.tight_layout()
plt.show()
```

---

### 2.4 Interpretation of scatter plot *(4 marks)*

```python
print("""
From the scatter plot the following can be deduced about medication efficacy:

1. meds5 (the most prescribed medication) is effective for patients with a 
   HIGH Ratio_Na_K (sodium-to-potassium ratio above ~15), regardless of age. 
   This suggests Ratio_Na_K is a strong predictor for meds5.

2. meds4 appears to cluster for patients with a LOW-to-MEDIUM Ratio_Na_K 
   and tends to be prescribed to older patients.

3. meds1, meds2, and meds3 occupy the lower Ratio_Na_K range. 
   These medications are differentiated more by the patient's age 
   than by their Na_K ratio alone.

4. Overall, the Ratio_Na_K appears to be the most important feature in 
   determining which medication is effective — there is a clear 
   horizontal separation in the scatter plot between meds5 and the others.
""")
```

---

### 2.5 Record count per medication *(1 mark)*

```python
print(meds['Medication'].value_counts())
# Expected: meds5=91, meds4=54, meds1=23, meds3=16, meds2=16
```

---

### 2.6 Balance check — pie chart *(4 marks)*

```python
counts = meds['Medication'].value_counts()
pct = (counts / counts.sum() * 100).round(2)

plt.figure(figsize=(7, 7))
plt.pie(pct, labels=pct.index, autopct='%.2f%%')
plt.title('Percentage of Records per Medication')
plt.show()

# Expected: meds5≈45.5%, meds4≈27.0%, meds1≈11.5%, meds3≈8.0%, meds2≈8.0%
print("""
The data is clearly IMBALANCED. meds5 accounts for approximately 45.5% of all 
records, while meds2 and meds3 each account for only about 8%. This imbalance 
means a classifier could achieve high accuracy simply by always predicting meds5, 
without learning the actual distinguishing patterns. Oversampling is required.
""")
```

---

### 2.7 Oversample to balance the data *(6 marks)*

```python
from sklearn.utils import resample

# Find the count of the majority class
max_count = meds['Medication'].value_counts().max()
print(f"Target count per class: {max_count}")

# Oversample each minority class to match the majority class count
resampled_classes = []
for med in meds['Medication'].unique():
    subset = meds[meds['Medication'] == med]
    if len(subset) < max_count:
        # Oversample with replacement
        oversampled = resample(subset,
                               replace=True,
                               n_samples=max_count,
                               random_state=42)
        resampled_classes.append(oversampled)
    else:
        resampled_classes.append(subset)

# Combine and shuffle
meds_resampled = pd.concat(resampled_classes)
meds_resampled = meds_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Shape after resampling: {meds_resampled.shape}")
meds_resampled
```

---

### 2.8 Confirm successful resampling *(3 marks)*

```python
# Record count per medication
print(meds_resampled['Medication'].value_counts())
# Expected: all 5 medications = 91 records each → 455 total

# Updated pie chart
counts_r = meds_resampled['Medication'].value_counts()
pct_r = (counts_r / counts_r.sum() * 100).round(2)

plt.figure(figsize=(7, 7))
plt.pie(pct_r, labels=pct_r.index, autopct='%.2f%%')
plt.title('Medication Distribution After Resampling')
plt.show()
# Expected: all slices equal at 20.00%
```

---

### 2.9 Convert text values to numeric *(6 marks)*

```python
from sklearn.preprocessing import LabelEncoder

# Binary columns (2 unique values) → [0, 1]
# Gender: Female=0, Male=1 | Cholesterol: High=0, Normal=1
binary_cols = ['Gender', 'Cholesterol']
le = LabelEncoder()
for col in binary_cols:
    meds_resampled[col] = le.fit_transform(meds_resampled[col])
print("Binary encoded:", binary_cols)

# Multi-value columns (>2 unique text values) → [0, 1, 2, ..., n]
# NOTE: 2022 paper asks for label encoding (NOT one-hot) for multi-value columns
# Blood_Presure: 3 values → [0, 1, 2]
# Medication: 5 values → [0, 1, 2, 3, 4]
multi_cols = ['Blood_Presure', 'Medication']
for col in multi_cols:
    meds_resampled[col] = le.fit_transform(meds_resampled[col])
print("Label encoded:", multi_cols)

meds_resampled
```

> **⚠️ Important 2022-specific note:** Unlike 2023/2024/2025 papers which use one-hot encoding for multi-value columns, the **2022 paper explicitly instructs** encoding multi-value columns as `[0, 1, 2, …, n]` — this means **Label Encoding** for all text columns, not one-hot encoding.

---

### 2.10 Confirm no object columns remain *(2 marks)*

```python
# Check dtypes
print(meds_resampled.dtypes)

# Confirm: should show no 'object' type columns
object_cols = list(meds_resampled.select_dtypes(include=['object', 'string']).columns)
print(f"\nObject columns remaining: {object_cols}")
# Expected: [] — empty list
```

---

### 2.11 Correlation heatmap *(4 marks)*

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
corr = meds_resampled.corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap — Medicine Dataset')
plt.tight_layout()
plt.show()

# Correlation with Medication specifically
print("Correlation with Medication (sorted):")
print(corr['Medication'].sort_values(ascending=False))
```

---

### 2.12 Interpretation of correlation values *(2 marks)*

```python
print("""
From the correlation heatmap, the following can be deduced about which 
features most influence the medication prescribed:

1. Ratio_Na_K has the STRONGEST positive correlation with Medication 
   (~0.59). This means a patient's sodium-to-potassium ratio is the 
   most important predictor of which medication will be prescribed.

2. Blood_Presure shows a moderate positive correlation (~0.37), 
   indicating it also plays a significant role in the prescription decision.

3. Cholesterol and Gender show very weak correlation values (close to 0), 
   suggesting they have minimal influence on which medication is prescribed.

4. Age shows a near-zero or slightly negative correlation, meaning age 
   alone is not a strong predictor of the medication type.

Conclusion: Ratio_Na_K and Blood_Presure are the most influential 
features; Cholesterol, Gender, and Age contribute very little.
""")
```

---

### 2.13 Define X, y and train/test split *(8 marks)*

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# X = all features (drop target), y = Medication
X = meds_resampled.drop('Medication', axis=1)
y = meds_resampled['Medication']

print(f"X dimensions: {X.shape}")
print(f"y dimensions: {y.shape}")

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train dimensions: {X_train.shape}")
print(f"X_test dimensions:  {X_test.shape}")

# Feature scaling (essential for KNN; good practice for all)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

---

### 2.14 Train classifiers with k-fold cross-validation *(8 marks)*

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

classifiers = {
    'KNN':           KNeighborsClassifier(),          # default k-value
    'Decision Tree': DecisionTreeClassifier(),          # default values
    'Random Forest': RandomForestClassifier(n_estimators=5)  # 5 trees
}

k = 5
results = {}

for name, clf in classifiers.items():
    acc = cross_val_score(clf, X_train, y_train, cv=k, scoring='accuracy')
    f1  = cross_val_score(clf, X_train, y_train, cv=k, scoring='f1_weighted')
    results[name] = {'accuracy': acc.mean(), 'f1': f1.mean()}
    print(f"{name}:")
    print(f"  CV Training Accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"  CV F1 Score:          {f1.mean():.4f} ± {f1.std():.4f}\n")
```

---

### 2.15 Best model evaluation *(3 marks)*

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Select model with highest F1 score
best_name = max(results, key=lambda k: results[k]['f1'])
print(f"Best model (highest F1): {best_name}")

best_clf = classifiers[best_name]
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

### 2.16 Discuss model metrics *(4 marks)*

```python
print("""
=== Model Discussion ===

1. Training Accuracy vs Testing Accuracy:
   If the training accuracy (from cross-validation) is much higher than the 
   test accuracy, the model is OVERFITTING — it has memorised training data 
   but cannot generalise. If the values are close, the model generalises well.
   For a good model, training and test accuracy should be similar.

2. Precision:
   Precision = TP / (TP + FP).
   It answers: "Of all the times the model predicted a patient needs meds_X, 
   how often was it correct?" A high precision means few false prescriptions.

3. Recall:
   Recall = TP / (TP + FN).
   It answers: "Of all patients who actually need meds_X, how many did the 
   model correctly identify?" A high recall means few missed prescriptions — 
   critical in a medical context where missing a diagnosis can be harmful.

4. F1-Score:
   F1 = 2 × (Precision × Recall) / (Precision + Recall).
   The harmonic mean of precision and recall. Since there are 5 medication 
   classes, the weighted F1-score accounts for class imbalance and gives a 
   single overall performance measure. A high weighted F1 indicates the model 
   predicts all medication types accurately and consistently.
""")
```

---
