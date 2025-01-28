_ = """
This file is used to build an ANN model using the Annoy library and the word2vec model.
"""

from annoy import AnnoyIndex
import gensim.downloader as api
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# Text preprocessing function, including tokenization and stop word removal
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    stop_words = set(stopwords.words('english'))  # Stop words
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Compute text similarity function
def compute_similarity(text1, text2, model):
    # Preprocess text
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)

    # Compute text vector representations
    vector1 = sum(model.get_vector(token) for token in tokens1 if token in model.key_to_index)
    vector2 = sum(model.get_vector(token) for token in tokens2 if token in model.key_to_index)
    # vector1 = np.atleast_2d(vector1)
    # vector2 = np.atleast_2d(vector2)
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray or len(vector1) != len(vector2):
        return -1
    # Compute cosine similarity
    similarity_score = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    return similarity_score

def str_to_vector(vector_str):
    clean_str = vector_str.strip("[]")
    clean_str = clean_str.replace("\n", "")
    str_list = clean_str.split()
    return [float(num) for num in str_list]

def update_ann_model(schema_name, config):
    embedding_path = os.path.join("./data/embedding", f"{schema_name}.csv")
    ann_path = os.path.join("./data/ann", f"{schema_name}.ann")

    df = pd.read_csv(embedding_path)
    vec_dim = config['ann_model']['dimensions']
    tree = AnnoyIndex(vec_dim, 'angular')

    for index, row in df.iterrows():
        vec_str = row['vector']
        vector = str_to_vector(vec_str)
        if len(vector)!= vec_dim:
            vector = [0] * vec_dim
        tree.add_item(row['vector_index'], vector)
    
    os.makedirs(os.path.dirname(ann_path), exist_ok=True)

    tree.build(config['ann_model']['tree_num'])
    tree.save(ann_path)