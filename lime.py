import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np

# Loading the training dataset
df = pd.read_csv('train.csv')
#Cutting down data to 1000 rows to perform initial assetments
data = df.head(1000)
# Step 2: Split the data into train and test sets
X = data['medical_abstract']
y = data['condition_label']
# Creating training and testing sets from the datasets. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the text classification pipeline with TFIDF and LR
pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

# Training the model
pipeline.fit(X_train, y_train)

# Instantiate the LIME Text Explainer
class_names = y.unique()
explainer = LimeTextExplainer(class_names=class_names)

# Explaination of model prediction
# Selecting a sample sentence from the test set
sample_sentence = X_test.iloc[0]
print(f"Explaining prediction for: '{sample_sentence}'")

# Predicting the class probabilities with words highlights
exp = explainer.explain_instance(
    sample_sentence,
    pipeline.predict_proba,
    num_features=9,
    top_labels=1
)

# Display the explanation
exp.show_in_notebook()
