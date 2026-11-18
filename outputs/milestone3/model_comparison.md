| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time (10 trips) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Linear & Probabilistic | StandardScaler + L2 C=1.0, lbfgs, max_iter=500 | 0.8846 ± 0.0169 | 0.8437 | 0.9300 | 0.8943 | 0.1258 | 0.006s (0.6 ms/trip) |
| Decision Tree | Tree-based | CART Gini, max_depth=12, min_samples_leaf=50 | 0.8983 ± 0.0190 | 0.8449 | 0.9259 | 0.9083 | 0.1061 | 0.008s (0.8 ms/trip) |
| Random Forest | Ensemble (Bagging) | 150 trees, max_depth=16, min_samples_leaf=20, max_features=sqrt | 0.9032 ± 0.0184 | 0.8528 | 0.9295 | 0.9127 | 0.0969 | 0.471s (47.1 ms/trip) |
| Histogram Gradient Boosting | Boosting | max_iter=200, lr=0.08, max_leaf_nodes=63, min_samples_leaf=50, l2=1.0 | 0.9035 ± 0.0185 | 0.8548 | 0.9362 | 0.9134 | 0.0962 | 0.337s (33.7 ms/trip) |
| K-Nearest Neighbours | Instance-based | StandardScaler + k=25, distance-weighted Euclidean (brute BLAS, batched) | 0.8927 ± 0.0179 | 0.8446 | 0.9279 | 0.9048 | 0.1503 | 111.179s (11117.9 ms/trip) |
