# Emotion Classification using NLP

## Project Overview

This project performs Emotion Classification on text data using Natural Language Processing (NLP) techniques. The dataset is loaded from `train.txt`, preprocessed through multiple NLP cleaning steps, converted into numerical features, and classified using Machine Learning algorithms.

---

## Dataset

The dataset contains two columns:

* **text** → Input sentence/text
* **emotion** → Corresponding emotion label

Example:

| text                     | emotion |
| ------------------------ | ------- |
| i am feeling happy today | joy     |
| i feel very sad          | sadness |
| i am scared of exams     | fear    |

---

## Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## NLTK Resources

Download required NLTK resources:

```python
nltk.download('punkt')
nltk.download('stopwords')
```

---

## NLP Preprocessing Steps

### 1. Lowercase Conversion

Convert all text into lowercase.

Example:

```text
I LOVE NLP
```

becomes

```text
i love nlp
```

---

### 2. Remove Punctuation

Remove symbols such as:

```text
.,!?;:"'()
```

Example:

```text
hello!!!
```

becomes

```text
hello
```

---

### 3. Remove Numbers

Remove numeric values from text.

Example:

```text
i have 5 books
```

becomes

```text
i have books
```

---

### 4. Remove Emojis

Remove emojis and non-ASCII characters.

Example:

```text
i am happy 😊
```

becomes

```text
i am happy
```

---

### 5. Tokenization

Split text into words using NLTK.

Example:

```text
i love machine learning
```

becomes

```python
['i', 'love', 'machine', 'learning']
```

---

### 6. Stopword Removal

Remove common words that do not carry significant meaning.

Examples:

```text
the
is
am
are
was
were
and
```

Example:

```text
i am learning nlp
```

becomes

```text
learning nlp
```

---

## Feature Extraction

### Count Vectorizer

Converts text into word frequency vectors.

Example:

| Word  | Count |
| ----- | ----- |
| happy | 2     |
| sad   | 1     |
| joy   | 3     |

---

### TF-IDF Vectorizer

Calculates the importance of a word in a document relative to the entire corpus.

Advantages:

* Reduces importance of common words
* Highlights informative words
* Often improves classification accuracy

---

## Machine Learning Models

### 1. Multinomial Naive Bayes

Suitable for text classification tasks.

Advantages:

* Fast
* Efficient
* Works well on sparse text data

---

### 2. Logistic Regression

A linear classification algorithm that often provides strong performance for NLP tasks.

Advantages:

* High accuracy
* Easy to interpret
* Effective on large datasets

---

## Workflow

```text
train.txt
    │
    ▼
Load Dataset
    │
    ▼
Lowercase
    │
    ▼
Remove Punctuation
    │
    ▼
Remove Numbers
    │
    ▼
Remove Emojis
    │
    ▼
Tokenization
    │
    ▼
Stopword Removal
    │
    ▼
CountVectorizer / TF-IDF
    │
    ▼
Train-Test Split
    │
    ▼
Naive Bayes
    │
    ▼
Logistic Regression
    │
    ▼
Accuracy Evaluation
```

---

## Evaluation Metric

### Accuracy Score

Formula:

```text
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

Higher accuracy indicates better model performance.

---

## Project Structure

```text
Emotion-Classification/
│
├── train.txt
├── emotion_classification.ipynb
├── README.md
│
└── requirements.txt
```

---

## Expected Outcome

The model predicts emotions such as:

* Joy
* Sadness
* Anger
* Fear
* Love
* Surprise

from user-provided text after NLP preprocessing and feature extraction.

---

## Future Improvements

* Lemmatization
* Stemming
* Hyperparameter tuning
* Cross-validation
* Deep Learning (RNN, LSTM, GRU)
* Transformer-based models (BERT, RoBERTa)
* Generative AI and Large Language Models

---

## Author

Nikhil Dhaduk

NLP | Machine Learning | Deep Learning | Generative AI
