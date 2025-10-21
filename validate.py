import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

test_data = pd.read_csv('test.csv')
X_test = test_data.drop('Survive', axis=1)
y_test = test_data['Survive']

y_pred = model.predict(X_test)

metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred, average='weighted')),
    'recall': float(recall_score(y_test, y_pred, average='weighted')),
    'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to metrics.json:")
print(json.dumps(metrics, indent=2))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to confusion_matrix.png")