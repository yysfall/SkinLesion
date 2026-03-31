import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("HAM10000_metadata.csv")
df.columns = df.columns.str.strip()

df["age"] = df["age"].fillna("").astype(str)
df["localization"] = df["localization"].fillna("").astype(str)
df["sex"] = df["sex"].fillna("").astype(str)

df["text"] = df["age"] + " " + df["sex"] + " " + df["localization"]
df["label"] = df["dx"].apply(lambda x: 1 if str(x).lower() == "mel" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)

print(classification_report(y_test, y_pred, digits=4))
print("Sample NLP prediction:", clf.predict(vectorizer.transform(["60 male scalp"]))[0])