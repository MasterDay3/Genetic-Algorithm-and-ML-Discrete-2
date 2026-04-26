DATA
→ load dataset (CSV using pandas)
→ initial data inspection
→ check missing values and duplicates
→ analyze target distribution (class balance)
→ identify feature types (numerical / categorical / binary)

PREPROCESSING
→ split into X (features) and y (target)
→ one-hot encoding for categorical features
→ handle missing values (if any)
→ feature scaling (StandardScaler for numerical features)
→ train/test split with stratification (preserve class balance)
→ prepare final feature matrix for modeling

BASELINE ML
→ choose baseline model
→ train model on all features (X_train)
→ make predictions on X_test
→ compute evaluation metrics
→ store results as baseline reference

GA (ML inside fitness)
→ encode solution as binary chromosome (1 = select feature, 0 = ignore)
→ initialize random population of feature subsets
→ fitness evaluation for each chromosome:
 → select subset of features from X_train
 → train ML model on selected features
 → evaluate using cross-validation
 → apply penalty for number of features
→ selection (choose best-performing individuals)
→ crossover (combine parent chromosomes)
→ mutation (random bit flips for diversity)
→ repeat over multiple generations
→ obtain best chromosome (optimal feature subset)

BEST FEATURES
→ decode best chromosome into selected feature list
→ interpret most important features
→ reduce dimensionality of dataset
→ create X_train_selected and X_test_selected

FINAL ML
→ train same ML model on selected features
→ make predictions on test set
→ compute final evaluation metrics

EVALUATION
→ evaluate final model performance (Accuracy, F1-score, AUC)
→ check generalization (train vs test performance gap)
→ assess stability of results

COMPARISON
→ compare baseline ML vs GA-optimized ML
→ analyze performance difference (AUC/F1)
→ analyze feature reduction impact
→ draw conclusion on effectiveness of GA-based feature selection
