import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer


ps=PorterStemmer()
punc=string.punctuation
sw=stopwords.words('english')

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
   # text=y cant be copied like this
    text=y[:]
    y.clear()
    for i in text:
        if i not in punc and i not in sw:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y) # list to string (printing )

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))


st.title(" Email/SMS Spam Classifier ")

inp=st.text_area('Enter the message ')
if st.button('Predict'):
    # 1.preprocessing
    trans_sms = transform_text(inp)
    # 2. vectorize
    vector_inp = tfidf.transform([trans_sms])
    # 3. Predict
    result = model.predict(vector_inp)[0]

    # 4.Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")




