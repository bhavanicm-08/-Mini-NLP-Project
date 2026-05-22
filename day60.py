from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("===================================")
print("     NLP TEXT CLASSIFICATION")
print("===================================")

texts = [
    "Win a free mobile now",
    "Claim your cash prize",
    "Congratulations you won lottery",
    "Hello how are you",
    "Please submit the assignment",
    "Let us meet tomorrow",
    "Are you coming to class"
]

labels = [1, 1, 1, 0, 0, 0, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

test_messages = [
    "Win free cash now",
    "Please attend the class today"
]

X_test = vectorizer.transform(test_messages)

predictions = model.predict(X_test)

for msg, pred in zip(test_messages, predictions):

    if pred == 1:
        print("\nMessage:", msg)
        print("Prediction: Spam")

    else:
        print("\nMessage:", msg)
        print("Prediction: Not Spam")