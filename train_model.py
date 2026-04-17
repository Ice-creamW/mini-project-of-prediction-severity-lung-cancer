import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('cancer patient data sets(1).csv')

X = df.drop(['index', 'Patient Id', 'Level'], axis=1)
y = df['Level']

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

