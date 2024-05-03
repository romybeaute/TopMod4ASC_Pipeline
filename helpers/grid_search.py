import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from bertopic import BERTopic
from umap import UMAP
import hdbscan
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import itertools

from helpers.BERT_helpers import hyperparams


def calculate_coherence(model, data):
    tokens = [doc.split() for doc in data]  # Simple whitespace tokenizer
    dictionary = Dictionary(tokens)

    # Debug: Print DataFrame columns to check structure
    topic_freq_df = model.get_topic_freq()
    print("DataFrame Columns:", topic_freq_df.columns)

    # topics = [
    #     [dictionary.token2id[word] for word, _ in model.get_topic(topic_id) if word in dictionary.token2id]
    #     for topic_id in topic_freq_df.Topic.values]
    # corpus = [dictionary.doc2bow(doc) for doc in tokens]
    topics = [[word for word, _ in model.get_topic(topic_id)] for topic_id in model.get_topics()]

    coherence_model = CoherenceModel(topics=topics, texts=tokens, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()  # Return coherence score

# def calculate_coherence(model, data):
#     # Debugging to see output of get_topic for each topic_id
#     tokens = [doc.split() for doc in data]  # Simple whitespace tokenizer

#     dictionary = Dictionary(tokens)
#     # corpus = [dictionary.doc2bow(doc) for doc in tokens]
#     topics = [[word for word, _ in model.get_topic(topic_id)] for topic_id in model.get_topics()]

#     coherence_model = CoherenceModel(topics=topics, texts=tokens, dictionary=dictionary, coherence='c_v')
#     return coherence_model.get_coherence() # Return coherence score

    


class BERTopicGridSearch(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5, n_neighbors=15, min_dist=0.1, min_cluster_size=5, min_samples=5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model = None

    def fit(self, X, y=None):
        umap_model = UMAP(n_components=self.n_components, n_neighbors=self.n_neighbors, min_dist=self.min_dist, random_state=42)
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, prediction_data=True)
        self.model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, nr_topics="auto")
        self.model.fit(X)
        return self

    def score(self, X, y=None):
        topics = [self.model.get_topic(topic_idx) for topic_idx in range(len(self.model.get_topics()))]
        texts = [doc.split() for doc in X]
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        coherence_model = CoherenceModel(topics=topics, texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_v')
        return coherence_model.get_coherence()

def get_params_grid(len_dataset):
    hyperparams_dict = hyperparams(len_dataset)
    hyperparams_umap = hyperparams_dict['umap_params']
    hyperparams_hdbscan = hyperparams_dict['hdbscan_params']
    #print hyperparameters for UMAP and HDBSCAN
    print("Hyperparameters for UMAP: ", hyperparams_umap)
    print("Hyperparameters for HDBSCAN: ", hyperparams_hdbscan)

    # Define the grid search
    param_grid = list(itertools.product(hyperparams_umap['n_neighbors'],
                                    hyperparams_umap['n_components'],
                                    hyperparams_umap['min_dist'],
                                    hyperparams_hdbscan['min_cluster_size'],
                                    hyperparams_hdbscan['min_samples']))
    print("Number of combinations: ", len(param_grid))
    return hyperparams_umap, hyperparams_hdbscan, param_grid


