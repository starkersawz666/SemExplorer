_ = """
This file contains the code for computing text vector representations using word2vec model.
Run this file to generate a new dataset with text vector representations once you change the word2vec model or the dataset.
"""

import gensim.downloader as api
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import os
import json
import streamlit as st

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
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray or len(vector1) != len(vector2):
        return -1
    # Compute cosine similarity
    similarity_score = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    return similarity_score

def update_embedding(schema_name, model):
    dataset_path = os.path.join("./data/info", f"{schema_name}.csv")
    schema_path = os.path.join("./data/schema", f"{schema_name}.json")
    saving_path = os.path.join("./data/embedding", f"{schema_name}.csv")

    df = pd.read_csv(dataset_path)
    df['merged_text'] = ''
    schema_map = json.load(open(schema_path))
    semantic_text_cols = schema_map['Text-semantic']
    for index, row in df.iterrows():
        merged_text = ''
        for col in semantic_text_cols:
            if col in row and not pd.isnull(row[col]):
                merged_text += row[col] + '; '
        df.at[index,'merged_text'] = merged_text
    
    cnt = 0
    df['vector'] = None
    df['vector_index'] = None
    for index, row in df.iterrows():
        tokens = preprocess_text(row['merged_text'])
        vector = sum(model.get_vector(token) for token in tokens if token in model.key_to_index)
        df.at[index,'vector'] = vector
        df.at[index, 'vector_index'] = cnt
        cnt += 1
    
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)
    df.to_csv(saving_path, index=False)

    st.success(f"Embedding for '{schema_name}' have been successfully updated.")