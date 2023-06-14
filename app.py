import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

port_stem = PorterStemmer()
vect = TfidfVectorizer()

# Load the pickle files
vect_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))


# Function to remove stopwords
def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con


# Define function to classify news
def news_classif(news):
    news = stemming(news)
    input_data = [news]
    # convert to a TfVectorized
    vector_form = vect_form.transform(input_data)
    # predict vector_form
    prediction = load_model.predict(vector_form)
    return prediction


if __name__ == '__main__':
    st.title('Fake News Classification app')
    st.subheader('Input the News content you want to classify below')
    sentence = st.text_area('Enter your news content here', 'Some news', height=200)
    predict_btn = st.button('Classify')
    if predict_btn:
        prediction_class = news_classif(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.warning("This is Unreliable news. I recommend that you don't trust it!")
        if prediction_class == [1]:
            st.success("This is Reliable news!")
