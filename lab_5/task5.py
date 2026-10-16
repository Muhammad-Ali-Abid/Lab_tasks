import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# from prettytable import PrettyTable  # Not needed while prediction is commented
# from astropy.table import Table, Column  # Unused; optional package

sample_data = pd.read_csv("sample-data.csv")

print("\n\nSample Data:")
print("============\n")
pd.set_option("display.max_rows", None, "display.max_columns", None)
# print(sample_data)

print("\n\nAttributes in Sample Data:")
print("==========================\n")

print(sample_data.columns)

print("\n\nNumber of Instances in Sample Data:",sample_data["PClass"].count())
print("========================================\n")  

pclass = pd.DataFrame({"Pclass":["First","Second","Third"]})
gender = pd.DataFrame({"Gender":["Male","Female"]})
sibling = pd.DataFrame({"Sibling":["Zero","One","Two","Three"]})
embarked = pd.DataFrame({"Embarked":["Southampton","Cherbourg","Queenstown"]})
survived = pd.DataFrame({"Survived":["Yes","No"]})

pclass_label_encoder = LabelEncoder()
gender_label_encoder = LabelEncoder()
sibling_label_encoder = LabelEncoder()
embarked_label_encoder = LabelEncoder()
survived_label_encoder = LabelEncoder()

pclass_label_encoder.fit(np.ravel(pclass))
gender_label_encoder.fit(np.ravel(gender))
sibling_label_encoder.fit(np.ravel(sibling))
embarked_label_encoder.fit(np.ravel(embarked))
survived_label_encoder.fit(np.ravel(survived))

with open("pclass_encoder.pkl", "wb") as f:
    pickle.dump(pclass_label_encoder, f)

with open("gender_encoder.pkl", "wb") as f:
    pickle.dump(gender_label_encoder, f)

with open("sibling_encoder.pkl", "wb") as f:
    pickle.dump(sibling_label_encoder, f)

with open("embarked_encoder.pkl", "wb") as f:
    pickle.dump(embarked_label_encoder, f)

with open("survived_encoder.pkl", "wb") as f:
    pickle.dump(survived_label_encoder, f)


sample_data_encoded_output = sample_data.copy()
original_sample_data = sample_data.copy()

print("\n\nSurvived Attribute After Label Encoding:")
print("========================================\n")
sample_data["encoded_survived"] = survived_label_encoder.transform(sample_data['Survived'])
print(sample_data[["Survived", "encoded_survived"]])

sample_data_encoded_output[['PClass', 'Gender', 'Sibling', 'Embarked', 'Survived']] = sample_data[['PClass', 'Gender', 'Sibling', 'Embarked', 'encoded_survived']]
pd.set_option("display.max_rows", None, "display.max_columns", None)
print("\n\nOriginal Sample Data:")
print("=====================\n")
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(original_sample_data)
print("\n\nSample Data after Label Encoding of Output:")
print("===========================================\n")
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(sample_data_encoded_output)

sample_data_encoded_output.to_csv(r'sample-data-encoded-output.csv', index = False, header = True)

sample_data_encoded = sample_data_encoded_output.copy()
sample_data_encoded_output_orignal = sample_data_encoded_output.copy()

print("\n\nPClass Attribute After Label Encoding:")
print("======================================\n")
sample_data_encoded_output["encoded_pclass"] = pclass_label_encoder.transform(sample_data_encoded_output['PClass'])
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(sample_data_encoded_output[["PClass", "encoded_pclass"]])

print("\n\nGender Attribute After Label Encoding:")
print("======================================\n")
sample_data_encoded_output["encoded_gender"] = gender_label_encoder.transform(sample_data_encoded_output['Gender'])
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(sample_data_encoded_output[["Gender", "encoded_gender"]])

print("\n\nSibling Attribute After Label Encoding:")
print("=======================================\n")
sample_data_encoded_output["encoded_sibling"] = sibling_label_encoder.transform(sample_data_encoded_output['Sibling'])
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(sample_data_encoded_output[["Sibling", "encoded_sibling"]])

print("\n\nEmbarked Attribute After Label Encoding:")
print("========================================\n")
sample_data_encoded_output["encoded_embarked"] = embarked_label_encoder.transform(sample_data_encoded_output['Embarked'])
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(sample_data_encoded_output[["Embarked", "encoded_embarked"]])

sample_data_encoded[['PClass', 'Gender', 'Sibling', 'Embarked', 'Survived']] = sample_data_encoded_output[['encoded_pclass', 'encoded_gender', 'encoded_sibling', 'encoded_embarked', 'Survived']]
print("\n\nOriginal Sample Data:")
print("=====================\n")
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(original_sample_data)
print("\n\nSample Data after Label Encoding:")
print("=================================\n")
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(sample_data_encoded)

sample_data_encoded.to_csv(r'sample-data-encoded.csv', index = False, header = True)

training_data_encoded, testing_data_encoded = train_test_split( sample_data_encoded , test_size=0.2 , random_state=42, shuffle = False)

# Save the Training and Testing Data into CSV File 

training_data_encoded.to_csv(r'training-data-encoded.csv', index = False, header = True)
testing_data_encoded.to_csv(r'testing-data-encoded.csv', index = False, header = True)

# print Training and Testing Data

print("\n\nTraining Data:")
print("==============\n")
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(training_data_encoded)
print("\n\nTesting Data:")
print("==============\n")
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(testing_data_encoded)

print("\n\nInputs Vectors (Feature Vectors) of Training Data:")
print("==================================================\n")

# Possible names for the original/encoded target columns (add other variants if you use them)
target_candidates = ["Survived", "Survived_encoded", "encoded_survived"]

# Determine which target column to use as the label (prefer encoded names first)
label_col = None
for name in ("Survived_encoded", "encoded_survived", "Survived"):
    if name in training_data_encoded.columns:
        label_col = name
        break

# Build a list of exactly two columns to drop:
# - If we found an encoded label (Survived_encoded / encoded_survived) drop both that and 'Survived' if present
# - If only 'Survived' exists, drop it (and any other explicit variant found)
cols_to_drop = []
if label_col:
    cols_to_drop.append(label_col)
# Also try to drop the plain 'Survived' column (if it's not already included)
if "Survived" in training_data_encoded.columns and "Survived" not in cols_to_drop:
    cols_to_drop.append("Survived")

# If you want to strictly ensure exactly two columns are dropped but only one was found, you can extend this list
# with any other known variant or simply rely on errors='ignore' below.
input_vector_train = training_data_encoded.drop(columns=cols_to_drop, errors='ignore')

print(f"Dropped columns: {cols_to_drop}")
print(input_vector_train)

print("\n\nOutputs/Labels of Training Data:")
print("================================\n")
if label_col:
    output_label_train = training_data_encoded[label_col]
else:
    raise KeyError("No target column found. Expected one of: " + ", ".join(target_candidates))

print(output_label_train)

print("\n\nInputs Vectors (Feature Vectors) of Training Data:")
print("==================================================\n")
# input_vector_train = training_data_encoded.iloc[: , :-1]
input_vector_train = training_data_encoded.drop(columns=["Survived", "encoded_survived"], errors='ignore')

print(input_vector_train)

print("\n\nOutputs/Labels of Training Data:")
print("================================\n")
print("  Survived")
output_label_train = training_data_encoded.iloc[: ,-1]
print(output_label_train)

# print("\n\nTraining the Support Vector Classifier on Training Data")
# print("========================================================\n")
# print("\nParameters and their values:")
# print("============================\n")
# svc_model = svm.SVC(gamma='auto',random_state=0)
# svc_model.fit(input_vector_train,np.ravel(output_label_train))
# print(svc_model)

# pickle.dump(svc_model, open('svc_trained_model.pkl', 'wb'))

feature_sets = {
    "Original": ["PClass", "Gender", "Sibling", "Embarked"],
    "FamilySize": [
        "PClass", "Gender", "Sibling",
        "Embarked", "FamilySize"
    ],
    "Both": [
        "PClass", "Gender", "Sibling",
        "Embarked", "FamilySize", "IsAlone"
    ]
}

df = sample_data[
    ["PClass", "Gender", "Sibling", "Embarked", "Survived"]
].copy()

# Exercise 1: create engineered features before selecting feature sets
sibling_map = {"Zero": 0, "One": 1, "Two": 2, "Three": 3}
if not df["Sibling"].isin(sibling_map).all():
    raise ValueError("Unexpected category in Sibling column")

# No parent/child/spouse feature exists in this sample, so FamilySize
# is approximated as sibling count plus the passenger.
df["FamilySize"] = df["Sibling"].map(sibling_map).astype(int) + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Match the original notebook's sequential 80/20 split.
train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, shuffle=False
)
y = survived_label_encoder.transform(df["Survived"])



def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(gamma="auto")
    }

results = []
cv_results = []

best_model = None
best_features = None
best_name = None
best_cv = -1

for config, features in feature_sets.items():

    X = df[features].copy()

    # Manual preprocessing for comparison
    for col in ["PClass", "Gender", "Sibling", "Embarked"]:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    for name, model in get_models().items():

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="accuracy"
        )

        results.append({
            "Model": name,
            "Features": config,
            "Accuracy": accuracy
        })

        cv_results.append({
            "Model": name,
            "Features": config,
            "CV Mean": scores.mean(),
            "CV Std": scores.std()
        })

        if scores.mean() > best_cv:
            best_cv = scores.mean()
            best_model = model
            best_features = features
            best_name = name

results_df = pd.DataFrame(results)
cv_df = pd.DataFrame(cv_results)

print("\nTest Accuracy:")
print(results_df.pivot(
    index="Model", columns="Features", values="Accuracy"
).round(3))

print("\nAccuracy change compared with original features:")
accuracy_table = results_df.pivot(
    index="Model", columns="Features", values="Accuracy"
)
print(accuracy_table.sub(
    accuracy_table["Original"], axis=0
).round(3))

results_df.to_csv("comparison.csv", index=False)


# print("\n\nInputs Vectors (Feature Vectors) of Testing Data:")
# print("=================================================\n")
# # input_vector_test = testing_data_encoded.iloc[: , :-1]
# input_vector_test = testing_data_encoded.drop(columns=["Survived", "encoded_survived"], errors='ignore')
# print(input_vector_test)

# print("\n\nOutputs/Labels of Testing Data:")
# print("==============================\n")
# print("  Survived")
# output_label_test = testing_data_encoded.iloc[: ,-1]
# print(output_label_test)

# model = pickle.load(open('svc_trained_model.pkl', 'rb'))

# print(input_vector_test)
# model_predictions = model.predict(input_vector_test)
# testing_data_encoded.copy(deep=True)
# pd.options.mode.chained_assignment = None
# testing_data_encoded["Predictions"] = model_predictions

# # Save the Predictions into CSV File

# testing_data_encoded.to_csv(r'model-predictions.csv', index = False, header = True)

# model_predictions = testing_data_encoded 
# print("\n\nPredictions Returned by svc_trained_model:")
# print("==========================================\n")
# print(model_predictions)

# model_accuracy_score = accuracy_score(model_predictions["Survived"],model_predictions["Predictions"])

# print("\n\nAccuracy Score:")
# print("===============\n")
# print(round(model_accuracy_score,2))



# pclass_input = input("\nPlease enter PClass here (First,Second,Third) : ").strip()
# gender_input = input("\nPlease enter your Gender here (Male, Female) : ").strip()
# sibling_input = input("\nPlease enter your Sibling here (Zero,One,Two,Three) : ").strip()
# embarked_input = input("\nPlease enter Embarked here (Cherbourg,Southampton,Queenstown) : ").strip()

# user_input = pd.DataFrame({ 'PClass': [pclass_input],'Gender': [gender_input],'Sibling': [sibling_input],'Embarked': [embarked_input]})

# print("\n\nUser Input Feature Vector:")
# print("==========================\n")
# print(user_input)

# with open("pclass_encoder.pkl", "rb") as f:
#     pclass_label_encoder = pickle.load(f)
# with open("gender_encoder.pkl", "rb") as f:
#     gender_label_encoder = pickle.load(f)
# with open("sibling_encoder.pkl", "rb") as f:
#     sibling_label_encoder = pickle.load(f)
# with open("embarked_encoder.pkl", "rb") as f:
#     embarked_label_encoder = pickle.load(f)
# with open("survived_encoder.pkl", "rb") as f:   # optional for inverse transform later
#     survived_label_encoder = pickle.load(f)

# ENCODERS = {
#     "PClass": pclass_label_encoder,
#     "Gender": gender_label_encoder,
#     "Sibling": sibling_label_encoder,
#     "Embarked": embarked_label_encoder
# }

# # ---------- (Optional) load your trained model ----------
# # with open("model.pkl", "rb") as f:
# #     model = pickle.load(f)
# # If you don't have a saved model, set model = None
# model = None

# # ---------- Example user input (replace with real input) ----------
# # user_input = pd.DataFrame([{
# #     "PClass": "Second",
# #     "Gender": "Female",
# #     "Sibling": "One",
# #     "Embarked": "Queenstown"
# # }])
# # or if you already have user_input DataFrame, keep it.

# # ---------- Ensure target column isn't passed to model ----------
# if "Survived" in user_input.columns:
#     # drop target if present in user input
#     user_input = user_input.drop(columns=["Survived"])

# # ---------- Determine expected feature names ----------
# if model is not None and hasattr(model, "feature_names_in_"):
#     expected_features = list(model.feature_names_in_)
# else:
    
#     expected_features = ["PClass", "Gender", "Sibling", "Embarked"]  # adjust if your model used different names/order


# extra = [c for c in user_input.columns if c not in expected_features]
# if extra:
#     user_input = user_input.drop(columns=extra)

# # Add any missing expected columns
# missing = [c for c in expected_features if c not in user_input.columns]
# if missing:
#     # Option A: fill with NaN (or sensible default values)
#     for c in missing:
#         user_input[c] = np.nan
#     # Option B (safer): raise an error so client must provide them
#     # raise ValueError(f"Missing required features for prediction: {missing}")

# # Reorder columns to match expected order
# user_input = user_input[expected_features]

# def validate_encoder_values(encoder, series, name):
#     known = set(map(str, encoder.classes_))
#     seen = set(map(str, series.dropna().astype(str)))
#     unknown = list(seen - known)
#     if unknown:
#         raise ValueError(f"Unknown categories for {name}: {unknown}")

# for col, enc in ENCODERS.items():
#     if col in user_input.columns:
#         validate_encoder_values(enc, user_input[col], col)


# unseen_data_features = user_input.copy()

# if "PClass" in unseen_data_features.columns:
#     unseen_data_features["PClass"] = pclass_label_encoder.transform(unseen_data_features["PClass"].astype(str))
# if "Gender" in unseen_data_features.columns:
#     unseen_data_features["Gender"] = gender_label_encoder.transform(unseen_data_features["Gender"].astype(str))
# if "Sibling" in unseen_data_features.columns:
#     unseen_data_features["Sibling"] = sibling_label_encoder.transform(unseen_data_features["Sibling"].astype(str))
# if "Embarked" in unseen_data_features.columns:
#     unseen_data_features["Embarked"] = embarked_label_encoder.transform(unseen_data_features["Embarked"].astype(str))

# print("\nUser Input Feature Vector:")
# print(user_input)
# print("\nUser Input Encoded Feature Vector:")
# print(unseen_data_features)

# model = pickle.load(open('svc_trained_model.pkl', 'rb'))

# predicted_survival = model.predict(unseen_data_features)

# if(predicted_survival == 1): 
#     prediction = "SURVIVED"
# if(predicted_survival == 0):
#     prediction = "NOT SURVIVED"

# pretty_table = PrettyTable()
# pretty_table.add_column("       ** Prediction **       ",[prediction])
# print(pretty_table)

# ================= EXERCISE 2: ROC CURVE AND AUC =================
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

Path('figures').mkdir(exist_ok=True)

# Select the configuration with the highest training CV score.
# Refit on the full training partition before assessing the held-out test data.
roc_features = df[best_features].copy()
for col in ['PClass', 'Gender', 'Sibling', 'Embarked']:
    enc = LabelEncoder()
    roc_features[col] = enc.fit_transform(roc_features[col])

X_train_roc = roc_features.iloc[train_idx]
X_test_roc = roc_features.iloc[test_idx]
y_train_roc = y[train_idx]
y_test_roc = y[test_idx]

best_model.fit(X_train_roc, y_train_roc)
if hasattr(best_model, 'predict_proba'):
    roc_scores = best_model.predict_proba(X_test_roc)[:, 1]
else:
    roc_scores = best_model.decision_function(X_test_roc)

fpr, tpr, thresholds = roc_curve(y_test_roc, roc_scores)
roc_auc = roc_auc_score(y_test_roc, roc_scores)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'{best_name}: AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Best Cross-Validated Model')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curve.png', dpi=150)
plt.close()
print('\nEXERCISE 2: ROC / AUC')
print('Selected model:', best_name, '| Features:', best_features)
print('Test ROC AUC:', round(roc_auc, 3))
print('ROC figure saved to figures/roc_curve.png')


# ================= HOME TASK: LEAKAGE-SAFE PIPELINE =================
print('\nHOME TASK: ColumnTransformer + Pipeline + GridSearchCV')

pipeline_columns = ['PClass', 'Gender', 'Sibling', 'Embarked', 'FamilySize', 'IsAlone']
X_pipeline = df[pipeline_columns].copy()
X_pipeline_train = X_pipeline.iloc[train_idx]
X_pipeline_test = X_pipeline.iloc[test_idx]
y_pipeline_train = y[train_idx]
y_pipeline_test = y[test_idx]

categorical_cols = ['PClass', 'Gender', 'Sibling', 'Embarked']
numeric_cols = ['FamilySize', 'IsAlone']

preprocessing = ColumnTransformer([
    ('categorical', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols),
    ('numeric', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_cols)
])

model_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('classifier', SVC())
])

search = GridSearchCV(
    model_pipeline,
    param_grid={
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['linear', 'rbf']
    },
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
search.fit(X_pipeline_train, y_pipeline_train)
pipeline_test_accuracy = accuracy_score(
    y_pipeline_test, search.predict(X_pipeline_test)
)
joblib.dump(search.best_estimator_, 'titanic_best_model.joblib')
print('Best hyperparameters:', search.best_params_)
print('Pipeline mean CV accuracy:', round(search.best_score_, 3))
print('Pipeline test accuracy:', round(pipeline_test_accuracy, 3))
print('Manual best mean CV accuracy:', round(best_cv, 3))
print('Difference in CV accuracy:', round(search.best_score_ - best_cv, 3))
print('Saved model: titanic_best_model.joblib')


# ================= EXERCISE 5: WRITTEN REPORT =================
results_df.to_csv('comparison.csv', index=False)
cv_df.to_csv('cross_validation.csv', index=False)
accuracy_pivot = results_df.pivot(index='Model', columns='Features', values='Accuracy')
cv_pivot = cv_df.pivot(index='Model', columns='Features', values='CV Mean')
report = f'''# Titanic Passenger Survival Prediction — Lab Report

## Dataset and approach
- Samples: {len(df)}; training: {len(train_idx)}; testing: {len(test_idx)}.
- Original features: PClass, Gender, Sibling, Embarked.
- Engineered features: FamilySize = mapped sibling count + 1; IsAlone = 1 if FamilySize == 1.
- FamilySize is only a proxy because parents, children and spouse counts are not available.
- Train/test split: first 80% used for training, final 20% for testing (shuffle=False).

## Exercise 1 and 4: Test accuracy by algorithm and feature configuration
{accuracy_pivot.round(3).to_string()}

## Exercise 3: Five-fold mean training cross-validation accuracy
{cv_pivot.round(3).to_string()}

Best manual model by training CV: {best_name} using {best_features}, mean CV accuracy {best_cv:.3f}.

## Exercise 2: ROC evaluation
Held-out test ROC AUC: {roc_auc:.3f}. Figure saved to figures/roc_curve.png.

## Home assignment: Leakage-safe Pipeline
- Preprocessor uses most-frequent categorical imputation and OneHotEncoder for categorical features.
- Numeric features use median imputation and StandardScaler.
- GridSearchCV tunes SVC C, gamma and kernel using five-fold cross-validation.
- Best parameters: {search.best_params_}
- Pipeline mean CV accuracy: {search.best_score_:.3f}
- Pipeline held-out test accuracy: {pipeline_test_accuracy:.3f}
- Manual best mean CV accuracy: {best_cv:.3f}

## Interpretation and deployment recommendation
Use a fitted Pipeline instead of independently preprocessing the entire dataset before cross-validation.
The manual baseline is not a controlled leakage-only comparison, because its encoding and model/hyperparameters differ from the pipeline.
The small sample and ordered, non-shuffled split limit generalization claims. Validate with additional representative data before deployment.
The pipeline is saved in titanic_best_model.joblib and accepts the six raw feature columns used above.
'''
Path('report.md').write_text(report, encoding='utf-8')
print('\nEXERCISE 5: Report saved to report.md')
print('Cross-validation data saved to cross_validation.csv')
print('\nAll five in-lab exercises and the home task completed.')
