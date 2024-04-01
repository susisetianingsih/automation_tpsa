import pandas as pd
import numpy as np
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk

app = Flask(__name__)
CORS(app)

data = pd.read_csv("src/data_latih.csv")
tfidf_vectorizer = pickle.load(open("src/tfidf.pkl", 'rb'))
tfidf_matrix = tfidf_vectorizer.fit_transform(data['question'])

@app.route('/ask', methods=['POST'])
def ask():
    req_data = request.get_json()
    question = req_data['question']    
    query_vect = tfidf_vectorizer.transform([question])
    similarity = cosine_similarity(query_vect, tfidf_matrix)
    max_similarity = np.argmax(similarity, axis=None)
    
    closest_question = data.iloc[max_similarity]['question']
    similarity_score = similarity[0, max_similarity]
    answer = data.iloc[max_similarity]['answer']
    comment = data.iloc[max_similarity]['comment']
    
    hasil = {
        "closest_found" : closest_question,
        "similarity"    : similarity_score,
        "answer"        : answer,
        "comment"       : comment
    }
    
    return jsonify(hasil)

if __name__ == '__main__':
    app.run()