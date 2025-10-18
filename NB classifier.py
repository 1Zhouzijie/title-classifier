import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

with open('negative_trainingSet.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = [line.strip() for line in f if line.strip()]
negative_df = pd.DataFrame({'title': lines})

with open('positive_trainingSet.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = [line.strip() for line in f if line.strip()]
positive_df = pd.DataFrame({'title': lines})
positive_df['label'] = 1
negative_df['label'] = 0

train_df = pd.concat([positive_df, negative_df], ignore_index=True)


test_df = pd.read_csv('testSet-1000.csv')


vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_df['title'])
y_train = train_df['label']

X_test = vectorizer.transform(test_df['title given by manchine'].fillna(''))
y_test = test_df['Y/N'].map({'Y':1, 'N':0})


nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)


y_pred = nb_classifier.predict(X_test)


print("Naive Bayes Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))