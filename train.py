import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

train_data = pd.read_csv('train.csv')

X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

model = LogisticRegression(max_iter=1000, random_state=42)


model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
print(f"Model type: {type(model).__name__}")