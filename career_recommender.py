import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

# loading the data from csv
data = pd.read_csv('students.csv')
print("data loaded")
print(data.head())

# separating input and output
x = data.drop('recommendation', axis=1)
y = data['recommendation']

# converting text to numbers
enc = LabelEncoder()
y2 = enc.fit_transform(y)

# splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=0.2, random_state=42)
print("train size:", len(x_train))
print("test size:", len(x_test))
print("careers found:", list(enc.classes_))

# training decision tree
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(x_train, y_train)
print("dtree done")

# training random forest
rforest = RandomForestClassifier(n_estimators=50, random_state=42)
rforest.fit(x_train, y_train)
print("rforest done")

# checking accuracy
pred1 = dtree.predict(x_test)
pred2 = rforest.predict(x_test)

acc1 = accuracy_score(y_test, pred1)
acc2 = accuracy_score(y_test, pred2)

print("dtree accuracy:", round(acc1 * 100, 1), "%")
print("rforest accuracy:", round(acc2 * 100, 1), "%")

if acc2 >= acc1:
    final_model = rforest
    print("using random forest")
else:
    final_model = dtree
    print("using decision tree")

# testing with a sample student
sample = {
    'math_score': 85,
    'science_score': 80,
    'arts_score': 40,
    'commerce_score': 55,
    'logical_thinking': 1,
    'creativity': 0,
    'communication': 1,
    'technical_skills': 1,
    'likes_research': 1,
    'likes_helping': 0,
    'likes_business': 0,
    'risk_taker': 0,
}

s = pd.DataFrame([sample])
res = final_model.predict(s)
career = enc.inverse_transform(res)[0]

probs = final_model.predict_proba(s)[0]
results = pd.DataFrame({
    'career': enc.classes_,
    'confidence': (probs * 100).round(1)
}).sort_values('confidence', ascending=False)

print("------------------------------")
print("result:", career)
print(results.to_string(index=False))
print("------------------------------")

# plotting feature importance
cols = list(x.columns)
imp = final_model.feature_importances_

chart_data = pd.DataFrame({
    'feature': cols,
    'score': imp
}).sort_values('score', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(chart_data['feature'], chart_data['score'], color='steelblue')
plt.xlabel('score')
plt.title('feature importance chart')
plt.tight_layout()
plt.show()

# saving the model
with open('career_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(enc, f)

print("model saved")