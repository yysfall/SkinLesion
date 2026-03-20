import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("HAM10000_metadata.csv")


df['age'] = df['age'].fillna('').astype(str)
df['localization'] = df['localization'].fillna('')


df['text'] = df['age'] + " " + df['localization']


df['label'] = df['dx'].apply(lambda x: 1 if x.lower() in ['mel', 'melanoma'] else 0)


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']


clf = MultinomialNB()
clf.fit(X, y)

print("Sample NLP prediction:", clf.predict(vectorizer.transform(["30 back"])))