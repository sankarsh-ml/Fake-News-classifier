import numpy as np

def accuracy(X, y, vocab, W1, b1, W2, b2, W3, b3):
    preds = []
    for i in range(len(X)):
        z1 = X[i:i+1] @ W1 + b1
        A1 = np.maximum(0, z1)

        z2 = A1 @ W2 + b2
        A2 = np.maximum(0, z2)

        z3 = A2 @ W3 + b3
        prob = 1/(1+np.exp(-z3))
        preds.append(1 if prob >= 0.5 else 0)

    preds = np.array(preds)
    return np.mean(preds == y)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true==1) & (y_pred==1))
    tn = np.sum((y_true==0) & (y_pred==0))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))

    print("Confusion Matrix:")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")

