| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time (10 trips) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Linear & Probabilistic | StandardScaler + L2 C=1.0, lbfgs, max_iter=500 | 0.8307 ± 0.0241 | 0.9022 | 0.9489 | 0.9348 | 0.1121 | 0.001s (0.5 ms/trip) |
| K-Nearest Neighbours | Instance-based | StandardScaler + k=25, distance-weighted Euclidean (brute BLAS, batched) | 0.8123 ± 0.0079 | 0.9064 | 0.9452 | 0.9415 | 0.1273 | 2.400s (799.8 ms/trip) |
| Decision Tree | Tree-based | CART Gini, max_depth=12, min_samples_leaf=50 | 0.8308 ± 0.0214 | 0.8995 | 0.9384 | 0.9396 | 0.1201 | 0.001s (0.4 ms/trip) |
| Random Forest | Ensemble (Bagging) | 150 trees, max_depth=16, min_samples_leaf=20, max_features=sqrt | 0.8348 ± 0.0175 | 0.9030 | 0.9405 | 0.9435 | 0.0844 | 0.058s (19.3 ms/trip) |
| Histogram Gradient Boosting | Boosting | max_iter=200, lr=0.08, max_leaf_nodes=63, min_samples_leaf=50, l2=1.0 | 0.8218 ± 0.0017 | 0.9126 | 0.9523 | 0.9461 | 0.0833 | 0.053s (17.8 ms/trip) |
