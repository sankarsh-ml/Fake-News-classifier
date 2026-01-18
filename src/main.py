from preprocess import build_vocab, tf_vector, clean
from model import train_nn, final_classifier
from evaluate import accuracy, confusion_matrix
import pandas as pd
import numpy as np

fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")
true['label'] = 1
fake['label'] = 0

df = pd.concat([true,fake]).reset_index(drop = True)

df['content'] = (df['title'].fillna('')+' '+ df['text'].fillna('')).apply(clean)
df =  df[['content','label']]


vocab, word_ids = build_vocab(df['content'], vocab_size=10000)

X = np.array([tf_vector(t, vocab,word_ids) for t in df['content']])
y = np.array(df['label'])

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


W1, b1, W2, b2, W3, b3 = train_nn(X_train, y_train)

acc = accuracy(X_test, y_test, vocab, W1, b1, W2, b2, W3, b3)
print("Test Accuracy:", acc)

y_pred = []
for i in range(len(X_test)):
    z1 = X_test[i:i+1] @ W1 + b1
    A1 = np.maximum(0, z1)
    z2 = A1 @ W2 + b2
    A2 = np.maximum(0, z2)
    z3 = A2 @ W3 + b3
    prob = 1/(1+np.exp(-z3))
    y_pred.append(1 if prob >= 0.5 else 0)

y_pred = np.array(y_pred)
confusion_matrix(y_test, y_pred)


test_input = "WHO issues updated guidelines on emerging health risks"
print(final_classifier(test_input,W1,b1,W2,b2,W3,b3,vocab,word_ids))
