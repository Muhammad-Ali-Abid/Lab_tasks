# Titanic Passenger Survival Prediction — Lab Report

## Dataset and approach
- Samples: 100; training: 80; testing: 20.
- Original features: PClass, Gender, Sibling, Embarked.
- Engineered features: FamilySize = mapped sibling count + 1; IsAlone = 1 if FamilySize == 1.
- FamilySize is only a proxy because parents, children and spouse counts are not available.
- Train/test split: first 80% used for training, final 20% for testing (shuffle=False).

## Exercise 1 and 4: Test accuracy by algorithm and feature configuration
Features             Both  FamilySize  Original
Model                                          
Decision Tree        0.80        0.80      0.80
KNN                  0.85        0.85      0.80
Logistic Regression  0.75        0.75      0.75
Random Forest        0.80        0.80      0.80
SVC                  0.80        0.80      0.80

## Exercise 3: Five-fold mean training cross-validation accuracy
Features              Both  FamilySize  Original
Model                                           
Decision Tree        0.688       0.688     0.688
KNN                  0.625       0.625     0.675
Logistic Regression  0.688       0.688     0.712
Random Forest        0.612       0.612     0.638
SVC                  0.625       0.638     0.675

Best manual model by training CV: Logistic Regression using ['PClass', 'Gender', 'Sibling', 'Embarked'], mean CV accuracy 0.713.

## Exercise 2: ROC evaluation
Held-out test ROC AUC: 0.650. Figure saved to figures/roc_curve.png.

## Home assignment: Leakage-safe Pipeline
- Preprocessor uses most-frequent categorical imputation and OneHotEncoder for categorical features.
- Numeric features use median imputation and StandardScaler.
- GridSearchCV tunes SVC C, gamma and kernel using five-fold cross-validation.
- Best parameters: {'classifier__C': 0.1, 'classifier__gamma': 'scale', 'classifier__kernel': 'linear'}
- Pipeline mean CV accuracy: 0.738
- Pipeline held-out test accuracy: 0.750
- Manual best mean CV accuracy: 0.713

## Interpretation and deployment recommendation
Use a fitted Pipeline instead of independently preprocessing the entire dataset before cross-validation.
The manual baseline is not a controlled leakage-only comparison, because its encoding and model/hyperparameters differ from the pipeline.
The small sample and ordered, non-shuffled split limit generalization claims. Validate with additional representative data before deployment.
The pipeline is saved in titanic_best_model.joblib and accepts the six raw feature columns used above.
