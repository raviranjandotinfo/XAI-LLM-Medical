import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Loading the training and testing datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Ensuring that columns exist in the dataset
assert 'medical_abstract' in train_data.columns, "Column 'medical_abstract' not found in train.csv"
assert 'condition_label' in train_data.columns, "Column 'condition_label' not found in train.csv"
assert 'medical_abstract' in test_data.columns, "Column 'medical_abstract' not found in test.csv"
assert 'condition_label' in test_data.columns, "Column 'condition_label' not found in test.csv"

# Separating features and labels in both datasets
X_train = train_data['medical_abstract']
y_train = train_data['condition_label']
X_test = test_data['medical_abstract']
y_test = test_data['condition_label']

# Creating text vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Because it has more sentances 
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training logistic regression model with 1K iteration and 42 state
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluating the model 
y_pred = model.predict(X_test_tfidf)

# Print classification metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
