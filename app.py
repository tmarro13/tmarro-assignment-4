from flask import Flask, request, jsonify, send_file
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import re

app = Flask(__name__)

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(documents)

# Perform LSA
n_components = 100
svd = TruncatedSVD(n_components=n_components)
lsa_matrix = svd.fit_transform(tfidf_matrix)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def parse_document(doc):
    # Extract headers (assuming they're at the beginning and contain ':')
    headers = {}
    body = doc
    header_lines = re.findall(r'^(.*?):(.*)$', doc, re.MULTILINE)
    if header_lines:
        for key, value in header_lines:
            headers[key.strip()] = value.strip()
        # Remove headers from the body
        body = re.sub(r'^.*?:.*?$', '', doc, flags=re.MULTILINE).strip()
    
    return {
        'headers': headers,
        'body': body[:1000] + '...' if len(body) > 1000 else body  # Truncate body to 1000 characters
    }

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/styles.css')
def styles():
    return send_file('styles.css', mimetype='text/css')

@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    query_vec = vectorizer.transform([query])
    query_lsa = svd.transform(query_vec)

    similarities = [cosine_similarity(query_lsa[0], doc_lsa) for doc_lsa in lsa_matrix]
    top_indices = np.argsort(similarities)[::-1][:5]

    results = []
    for idx in top_indices:
        parsed_doc = parse_document(documents[idx])
        results.append({
            'document': parsed_doc,
            'similarity': float(similarities[idx])
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=3000)