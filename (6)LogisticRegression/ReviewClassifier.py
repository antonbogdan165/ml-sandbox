import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

movies = pd.read_csv("IMDB_Dataset.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(movies["review"])
y = movies["sentiment"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

probabilities = model.predict_proba(X)[:, 1]
movies["predictions"] = probabilities
movies_sorted = movies.sort_values(by="predictions")

most_negative = movies_sorted.iloc[0]["review"]
most_positive = movies_sorted.iloc[-1]["review"]

words = vectorizer.get_feature_names_out()
weights = model.coef_[0]
word_weights = pd.DataFrame({"word": words, "weight": weights})

print(f"Most negative: \n{most_negative}\n")
print(f"Most positive: \n{most_positive}\n")
print(f"Bias: {model.intercept_[0]}\n")
print(word_weights[word_weights["word"].isin(["wonderful", "horrible", "the"])])
