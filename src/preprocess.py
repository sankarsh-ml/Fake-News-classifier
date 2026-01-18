import pandas as pd
import numpy as np
import re
import string
from collections import Counter

def clean(text): #this is a function to clean up the text  remove the commas and stops etc
  if isinstance(text, str):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans("","",string.punctuation))
    text = re.sub(r"\s",' ', text).strip()
    return text
  return ''


def build_vocab(texts,vocab_size=10000):
  word_counts = Counter()
  for text in texts:
    words = text.split()

    word_counts.update(words)

    bigrams = [words[i] + " " + words[i+1] for i in range(len(words)-1)]
    word_counts.update(bigrams)

    trigrams = [words[i] + " " + words[i+1] + " " + words[i+2] for i in range(len(words)-2)]
    word_counts.update(trigrams)

  vocab = [w for w,c in word_counts.most_common(vocab_size)]
  word_ids = {w:i for i, w in enumerate(vocab)}
  return vocab, word_ids

def tf_vector(text,vocab,word_ids):#to make a vector of words with counts
  V = len(vocab)
  vec = np.zeros(V)
  words = text.split()

  for w in words:
    if w in word_ids:
      vec[word_ids[w]] +=1

  for i in range(len(words)-1):
    bg = words[i] + ' ' + words[i+1]
    if bg in word_ids:
      vec[word_ids[bg]] += 1
  return vec

