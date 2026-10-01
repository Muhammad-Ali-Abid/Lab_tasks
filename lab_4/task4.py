import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


housing = fetch_california_housing(as_frame=True)
df_reg = housing.frame

iris = load_iris(as_frame=True)
df_cls = iris.frame

print("Regression dataset shape:", df_reg.shape)
print("Classification dataset shape:", df_cls.shape)
# print(df_reg)
# print(df_cls)

X_reg = df_reg.drop(columns=['MedHouseVal'])
y_reg = df_reg['MedHouseVal']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

X_cls = df_cls.drop(columns=['target'])
y_cls = df_cls['target']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.25, random_state=42)
print("Training samples (Regression):", X_train_reg.shape[0])
print("Training samples (Classification):", X_train_cls.shape[0])

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)

print("R² Score:", r2_score(y_test_reg, y_pred_reg))

# log_model = LogisticRegression(max_iter=200)
log_model = DecisionTreeClassifier(random_state=42)
log_model.fit(X_train_cls, y_train_cls)

y_pred_cls = log_model.predict(X_test_cls)

print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_cls,y_pred_cls))
print("\nClassification Report:\n", classification_report(y_test_cls,y_pred_cls))

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline.fit(X_train_reg, y_train_reg)
y_pred_pipe = pipeline.predict(X_test_reg)
print("Pipeline R² Score:", r2_score(y_test_reg, y_pred_pipe))

import joblib

joblib.dump(reg_model, 'linear_model.pkl')
loaded_model = joblib.load('linear_model.pkl')

print("Loaded Model R²:", r2_score(y_test_reg,loaded_model.predict(X_test_reg)))

print()
print("Features\tCoefficients")
for feature, coefficient in zip(X_reg.columns, reg_model.coef_):
    print(f"{feature}  \t {coefficient}")

# Exercise 1: The plain Linear Regression and the StandardScaler + LinearRegression pipeline both achieve an R² of approximately 0.5758. Standardization does not materially change an ordinary least-squares Linear Regression model because rescaling an input feature can be compensated for by inversely rescaling its learned coefficient, resulting in essentially the same fitted predictions. Scaling is important for distance-based algorithms such as K-Nearest Neighbors, because differently scaled features would contribute unequally to distance calculations.

# Exercise 2: The two largest absolute coefficients of the unscaled Linear Regression model are approximately AveBedrms (+0.783) and MedInc (+0.449). However, raw coefficient magnitudes should not be treated as directly comparable measures of importance because each feature uses different units and numerical scales. Standardizing the features makes a one-unit change correspond to approximately one standard deviation, making coefficient magnitudes more meaningfully comparable.

# Exercise 3: With test_size=0.25, random_state=42, and LogisticRegression(max_iter=200), the model does not confuse any of the three species. The confusion matrix is perfectly diagonal: 15 setosa, 11 versicolor, and 12 virginica samples are classified correctly, with zero off-diagonal errors. This agrees with the classification report, where recall is 1.00 for all three classes.

# Exercise 4: Replacing Logistic Regression with DecisionTreeClassifier requires changing only the estimator construction; the .fit() and .predict() calls and the entire evaluation block remain valid. With the supplied Iris split, the Decision Tree also obtains 100% test accuracy. This demonstrates Scikit-learn's consistent estimator API, which allows algorithms to be exchanged without redesigning the surrounding machine-learning workflow.

df = pd.read_csv("two_source_corpus_features.csv")
print(df.columns)
median_polarity = df["sentiment_polarity"].median()

print("Median sentiment polarity:", median_polarity)

df["sentiment_class"] = (
    df["sentiment_polarity"] > median_polarity
).astype(int)

X = df["text"]
y = df["sentiment_class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

sentiment_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english"
        )
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

sentiment_pipeline.fit(
    X_train,
    y_train
)

y_pred = sentiment_pipeline.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

joblib.dump(
    sentiment_pipeline,
    "sentiment_pipeline.pkl"
)

print(
    "\nPipeline saved successfully."
)

model = joblib.load("sentiment_pipeline.pkl")

texts = [
    "The company reported strong growth, higher profits, and very positive customer feedback.",

    "The business suffered major losses, declining sales, and several serious customer complaints.",

    "The new product was successful and received excellent reviews from users.",

    "The project faced delays, technical problems, and disappointing results.",

    "The organization announced stable performance with moderate growth during the year.",

    "Customers were extremely satisfied with the improved service and faster response times."
]

predictions = model.predict(texts)

for i, j in zip(texts, predictions):
    print(f"{j} : {i}")