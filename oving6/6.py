
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import decomposition
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# |%%--%%| <R1CldXkhGc|YEJIAnIE3R>
r"""°°°
# Exercise 6

For this exercise you can use either Python with sklearn or Weka.

* Using the UCI mushroom dataset from the last exercise, perform a feature
selection using a classifier evaluator. Which features are most discriminitave?

* Use principal components analysis to construct a reduced space.
Which combination of features explain the most variance in the dataset?

* Do you see any overlap between the PCA features
and those obtained from feature selection?
°°°"""
# |%%--%%| <YEJIAnIE3R|7ew9k3J1dx>

df = pd.read_csv('agaricus-lepiota.csv')
df

df_dummies = pd.get_dummies(df).astype(float)

# Want to determine edibility, so it cannot be in the X dataset
X, y = df_dummies.drop(['edibility_e', 'edibility_p'],
                       axis=1), df_dummies[['edibility_e', 'edibility_p']]
print("X-shape: ", X.shape)

skb = SelectKBest(chi2, k=5)
skb.fit(X, y)
X_new = skb.transform(X)
print("X_new-shape: ", X_new.shape)

# Encodes True to the selected features out of the total 118
mask = skb.get_support(indices=True)
print("Selected column indices: ", mask)

selected_features = df_dummies.columns[mask]
# print("Selected columns: ", selected_features)
print("Selected features: ", ", ".join(
    selected_features.values))  # Joins on ", "

# |%%--%%| <7ew9k3J1dx|N21Af1Z3Cy>


# normalize data
data_scaled = pd.DataFrame(preprocessing.scale(X), columns=X.columns)

# PCA
# Common to reduce to 2D, probably because it is easy both to understand and to plot
pca = PCA(n_components=2)
x_pca = pca.fit_transform(data_scaled)

best_features = [pca.components_[i].argmax() for i in range(x_pca.shape[1])]
feature_names = [X.columns[best_features[i]] for i in range(x_pca.shape[1])]
print("Highest variance: ", feature_names)

# Dump components relations with features:
print(pd.DataFrame(pca.components_,
      columns=data_scaled.columns, index=['PC-1', 'PC-2']))

# |%%--%%| <N21Af1Z3Cy|OzYjIDkM2k>
r"""°°°
* Do you see any overlap between the PCA features
and those obtained from feature selection?
°°°"""
# |%%--%%| <OzYjIDkM2k|1ICGWMJKza>

set(selected_features).intersection(set(feature_names))
