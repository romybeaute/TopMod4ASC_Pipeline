"""
@Author : Romy Beauté
@Date creation : 30-04-2024
@Last modification : 02-05-2024
Contact : r.beaut@sussex.ac.uk
"""

import pandas as pd
import os 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import sys
import itertools


from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import CountVectorizer

# BERTopic Libraries
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

#import helpers functions manually created
sys.path.append('/Users/rb666/projects/TopicModelling_META/helpers')
project_path = os.path.abspath('/Users/rb666/projects/TopicModelling_META')
if project_path not in sys.path:
    sys.path.append(project_path)
from helpers.BERT_helpers import *
from helpers.get_DM_data import *
from helpers.grid_search import hyperparams, calculate_coherence,get_params_grid

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

####### DATA PARAMETERS SELECTION #######

#Select parameters for dataset and preprocessing
HighSensory = True #If False, corresponds to deep listening experience (no flicker)
Handwritten = False #If True, corresponds to handwritten reports




def perform_grid_search(data):
    # Get hyperparameters for UMAP and HDBSCAN (from BERT_helpers.py hyperparams function)
    hyperparams_umap, hyperparams_hdbscan, param_grid = get_params_grid(len(data))


    results = {}  # Dictionary to store results

    # Loop over the grid of parameters
    for params in param_grid:
        n_neighbors, n_components, min_dist, min_cluster_size, min_samples = params
        # try:
        print("Testing parameters: ", params)
        topics, model = main(data, *params)

        # Calculate coherence score
        coherence_score = calculate_coherence(model, data)
        print("Coherence Score: ", coherence_score)
        results[params] = float(coherence_score)


    # Store best results separately if needed
    best_score = max(results.values())
    best_params = max(results, key=results.get)

    print("Best Coherence Score:", best_score)
    print("Best Hyperparameters:", best_params)


    numeric_results = {k: v for k, v in results.items() if isinstance(v, float)} # Filter out non-numeric results

    # top_results = sorted(numeric_results.items(), key=lambda x: x[1].item() if hasattr(x[1], 'item') else x[1], reverse=True)[:5]
    top_results = sorted(numeric_results.items(), key=lambda x: x[1], reverse=True)[:5]

    print("Top 5 parameter combinations:")
    for i, (params, score) in enumerate(top_results, start=1):
        print(f"{i}. Parameters: {params}, Score: {score}")

    return top_results  







####### MAIN FUNCTION #######


def get_data(HighSensory, Handwritten):
    # ********** Load data **********
    reports = load_DM_data(HighSensory, Handwritten)
    # reports_df = reports.reset_index(drop=True).to_frame('reflection_answer')
    
    # ********** Pre-process data **********
    #Apply very basic pre-processing of data from clean_text function in BERT_helpers.py : (remove stopwords if set to True), punctuation, numbers, convert to lowercase
    reports_filtered = get_cleaned_data(reports,
                                        remove_stopwords=False,
                                        extended_stopwords=False,
                                        apply_vectorizer=False)
    return reports_filtered.tolist() #data


def main(data,n_neighbors, n_components, min_dist, min_cluster_size, min_samples):

    print(f"Received parameters: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}, min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    #will remove stopwords with the vectorizer model instead of the clean_text function
    stopwords_list = list(nltk_stopwords.words('english')) + list(custom_stopwords)
    print("Additional stopwords: ", custom_stopwords)
    
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords_list)

    # ********** Instanciate BERTOPIC **********

    #Embedding of reports using Sentence Transformer model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') #Loads pre-trained Sentence Transformer model 'all-MiniLM-L6-v2'
    
    umap_model = UMAP(n_neighbors=n_neighbors,
                      n_components=n_components,
                      min_dist=min_dist,
                      random_state=77) #set rdm seed for reportability
    
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, 
                            min_samples=min_samples,
                            gen_min_span_tree=True,
                            prediction_data=True) #set rdm seed for reportability
    
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=5,
        language='english',
        calculate_probabilities=True,
        verbose=True)

    # ********** Fit BERTOPIC **********
    topics, _ = model.fit_transform(data)


    return topics,model




if __name__ == "__main__":
    HighSensory = HighSensory  # If False, corresponds to deep listening experience (no flicker)
    Handwritten = Handwritten  # If True, corresponds to handwritten reports
    data = get_data(HighSensory, Handwritten)  
    perform_grid_search(data)