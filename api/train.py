
# # Local training

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

# ### Load the data

df = pd.read_csv("../data/mushrooms.csv")
df

# lable encoding

label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df

pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))

# ### Train/test split

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ### ensamble classifier

ensamble = VotingClassifier(estimators=[
    ('mnb', MultinomialNB()),
    ('svc', SVC()),
    ('rf', RandomForestClassifier())
])

# ### Optimize parameters

cls = GridSearchCV(
    ensamble, 
    {
        'mnb__alpha': [0.1, 1, 2],
        'svc__C': [0.1, 1, 10],
        'svc__class_weight': ['balanced'],
        'rf__n_estimators': [10, 100],
        'rf__criterion': ['gini', 'entropy'],
    },
    cv=5,
    scoring='f1_macro'
)


cls.fit(X_train, y_train)
cls.best_params_

# ### Print evaluation metrics

print('Validation score', cls.best_score_)
print('Test score', cls.score(X_test, y_test))

# ### Save the model

pickle.dump(cls, open('model.pkl', 'wb'))


