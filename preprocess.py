import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv("raw_data.csv")

df.drop(['Name', 'Cabin', 'Fare', 'Ticket', 'Embarked'], axis=1, inplace=True)

df['Age'] = df['Age'].fillna(df.Age.mean())

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

df['Child'] = df['Age'] < 18
df['Teen'] = (df['Age'] >= 18) & (df['Age'] < 25)
df['Adult'] = (df['Age'] >= 25) & (df['Age'] < 60)
df['Senior'] = df['Age'] >= 60
df.drop(['Age'], axis=1, inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,  
    stratify=y          
)

train_data = X_train.copy()
train_data['Survived'] = y_train

test_data = X_test.copy()
test_data['Survived'] = y_test

train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

print(f"Train set saved: {len(train_data)} rows")
print(f"Test set saved: {len(test_data)} rows")