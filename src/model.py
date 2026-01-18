import numpy as np
from preprocess import clean,tf_vector

def sigmoid(z):
  z = np.clip(z, -50, 50)
  return 1/(1+np.exp(-z))

def relu(z):
  return np.maximum(0,z)

def relu_deriv(z):
  return (z>0).astype(float)

def train_nn(X, y, lr=0.001, epochs=30, h1 = 128,h2=64,beta1=0.9,beta2=0.999,eps=1e-8):

    n,input_dim = X.shape
    W1 = np.random.randn(input_dim, h1)*np.sqrt(2/input_dim)
    b1 = np.zeros((1,h1))
    W2 = np.random.randn(h1,h2)*np.sqrt(2/h1)
    b2 = np.zeros((1,h2))
    W3 = np.random.randn(h2,1)*np.sqrt(2/h2)
    b3 = np.zeros((1,1))

    y = y.reshape(-1,1)

    params = [W1,b1,W2,b2,W3,b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    t = 0

    for epoch in range(epochs):
        t+=1

        z1 = X@W1 + b1
        A1 = relu(z1)

        z2 = A1@W2 + b2
        A2 = relu(z2)

        z3 = A2@W3 + b3
        A3 = sigmoid(z3)

        loss = -np.mean(y*np.log(A3 + eps)+(1-y)*np.log(1-A3 + eps))

        dz3 = A3 - y
        dW3 = A2.T@dz3/n
        db3 = np.mean(dz3, axis=0, keepdims=True)

        dA2 = dz3@W3.T
        dz2 = dA2*relu_deriv(z2)
        dW2 = A1.T@dz2/n
        db2 = np.mean(dz2, axis=0, keepdims=True)

        dA1 = dz2 @ W2.T
        dz1 = dA1 * relu_deriv(z1)
        dW1 = X.T @ dz1/n
        db1 = np.mean(dz1, axis=0, keepdims=True)

        grads = [dW1,db1,dW2,db2,dW3,db3]

        for i in range(len(params)):
          m[i] = beta1*m[i] + (1-beta1)*grads[i]
          v[i] = beta2*v[i] + (1-beta2)*(grads[i]**2)

          m_hat = m[i]/(1-beta1**t)
          v_hat = v[i]/(1-beta2**t)

          params[i] -= lr*m_hat/(np.sqrt(v_hat)+eps)

        W1, b1, W2, b2, W3, b3 = params


        print(f"epoch {epoch}: loss {loss}")

    return W1, b1, W2, b2, W3, b3

SENSATIONAL_WORDS = [
    "shocking", "secretly", "exposed", "leaked", "breaking",
    "unbelievable", "claims", "allegedly", "funds",
    "terrorist", "scandal", "insider", "reveals",
    "plot", "scheme", "agenda", "hidden",
    "sources", "anonymous", "secret", "coverup",
    "massive", "disaster", "destroy", "fraud", "rigged"
]

def sensationalism_score(tokens):
    return sum(w in SENSATIONAL_WORDS for w in tokens) / max(1, len(tokens))

ACCUSATION_VERBS = [
    "funds", "supports", "backs", "finances", "helped",
    "created", "runs", "leads", "controls"
]

ACCUSATION_OBJECTS = [
    "terrorist", "terrorists", "terrorism", "extremist", "extremists",
    "cartel", "criminal", "attack", "massacre"
]

def accusation_score(tokens):
    score = 0
    for i in range(len(tokens)-1):
        if tokens[i] in ACCUSATION_VERBS and tokens[i+1] in ACCUSATION_OBJECTS:
            score += 1
    return score

OFFICIAL_SOURCES = [
    "fbi", "cia", "police", "government", "officials",
    "investigators", "ministry", "department", "court",
    "report", "reports", "agency", "intelligence", "spokesperson"
]

def has_official_source(tokens):
    return any(w in OFFICIAL_SOURCES for w in tokens)

def style_score(text):
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / len(words)
    caps = sum(w.isupper() for w in words)

    score = 0
    if avg_word_len < 4: score += 1
    if caps > 0: score += 1
    return score

def predict(text,W1,b1,W2,b2,W3,b3,vocab,word_ids):
    text = clean(text)
    if len(text.split()) < 5:
        return "Input too short for reliable classification"

    vec = tf_vector(text,vocab,word_ids)
    vec = vec.reshape(1,-1)

    z1 = vec @ W1 + b1
    A1 = relu(z1)

    z2 = A1 @ W2 + b2
    A2 = relu(z2)

    z3 = A2 @ W3 + b3
    prob = float(sigmoid(z3))


    print(prob)
    return "FAKE NEWS" if prob <= 0.5 else "REAL NEWS"


def final_classifier(text,W1,b1,W2,b2,W3,b3,vocab,word_ids):
    tokens = clean(text)


    nn_label = predict(text,W1,b1,W2,b2,W3,b3,vocab,word_ids)


    sens = sensationalism_score(tokens)
    acc = accusation_score(tokens)
    official = has_official_source(tokens)
    style = style_score(text)


    if acc > 0 and not official:
        return "LIKELY FAKE (unsupported accusation)"

    if sens > 0.04 and nn_label == "REAL NEWS":
        return "LIKELY FAKE (sensational tone)"

    if style > 1 and nn_label == "REAL NEWS":
        return "LIKELY FAKE (dramatic writing)"

    if nn_label == "FAKE NEWS":
        return "FAKE NEWS"

    if acc > 0 and official:
        return "REAL NEWS (official accusation)"


    return nn_label
