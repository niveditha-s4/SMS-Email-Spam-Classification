# SMS-Email-Spam-Classification
Classifies SMS/EMail  messages as 'Spam' or 'Not Spam'.

# Email/SMS Spam Classifier using Streamlit

This is a simple web app built using Streamlit and Multinomial Naive Bayes to classify whether a given message is Spam or Not Spam. The model is trained on a labeled dataset of SMS messages and uses TF-IDF vectorization and text preprocessing techniques.

---

## Features

- Text preprocessing (tokenization, stopword removal, stemming)
- TF-IDF vectorization
- Accuracy and  precision comparision for 11 algorithms(SVC, NB , KN, RF , ETC, XGB, GBDT, DT, AdaBoost, LR, BGC)
- Spam detection using Multinomial Naive Bayes
- Interactive web interface built with Streamlit

---

## Model Details

- Model: Multinomial Naive Bayes
- Vectorizer: TF-IDF (3000 most frequent features)
- Dataset: SMS Spam Collection Dataset (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Preprocessing steps:
  - Lowercasing
  - Removing punctuation and stopwords
  - Stemming using NLTK

---


