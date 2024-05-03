#Handmade functions to help with BERT model
import re
from nltk.corpus import stopwords
from bertopic import BERTopic
import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm



# Additional stopwords can be added based on dataset-specific jargon
custom_stopwords = {'trevor', 'test','like','eg','devtest','hello','ture'}


def hyperparams(len_dataset):
    return {'umap_params': {
            'n_components': [2,3,5,7,10], #default to 2 
            'n_neighbors': [3,5,7,10,12],
            'min_dist': [0.0,0.01,0.025,0.05], #default to 1.0
        },
        'hdbscan_params': {
            #list of 3 values : 1% len_dataset,10 (default value), 5% len_dataset
            'min_cluster_size': [int(len_dataset*0.01),10,int(len_dataset*0.05)],
            'min_samples': [int(len_dataset*0.01),10,int(len_dataset*0.05),None] #default to None
        }
    }

# def hyperparams(len_dataset):
#     return {'umap_params': {
#             'n_components': [2], #default to 2 
#             'n_neighbors': [3],
#             'min_dist': [0.0,0.01], #default to 1.0
#         },
#         'hdbscan_params': {
#             #list of 3 values : 1% len_dataset,10 (default value), 5% len_dataset
#             'min_cluster_size': [int(len_dataset*0.01),],
#             'min_samples': [int(len_dataset*0.01)] #default to None
#         }
#     }



# Basic text cleaning and stopwords removal
def clean_text(text,remove_stopwords, stop_words=stopwords.words('english')):
    text = text.replace('\n', ' ') # remove newline characters
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.lower()  # convert to lowercase
    text = ' '.join(text.split())  # remove extra whitespaces

    if remove_stopwords:
        text = ' '.join([word for word in text.split() if word not in stop_words])
        #print the number of removed stopwords from english dictionnary and the number of stopwords from custom_stopwords and print the list of extended stopwords
        print(f"Number of stopwords removed: {len(set(text.split()).intersection(stop_words))}")
        print(f"Number of custom stopwords removed: {len(set(text.split()).intersection(custom_stopwords))}")
        print(f"Extended stopwords: {set(text.split()).intersection(custom_stopwords)}")

    return text







def customed_BERTopic(umap_model,hdbscan_model,embedding_model,vectorizer_model):
    model = BERTopic(
    n_gram_range=(1,2), 
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,
    language='english',
    calculate_probabilities=True,
    verbose=True)
    return model 

def base_model():
    return BERTopic(language="english", calculate_probabilities=True, verbose=True) #model with base parameters






def find_topics_c_TF_IDF(data, cluster_labels):
    # Tokenization
    tokenized_data = [nltk.tokenize.wordpunct_tokenize(sentence) for sentence in data]
    print(f"Number of tokenized documents: {len(tokenized_data)}")
    
    # Dataframe creation
    df = pd.DataFrame({'tokens': tokenized_data, 'class': cluster_labels})
    
    # Aggregate tokens by class
    class_aggregated_tokens = df.groupby('class')['tokens'].agg(sum)
    
    # Vocabulary creation
    all_tokens = [token for sublist in tokenized_data for token in sublist]  # Flatten the list
    vocab = list(set(all_tokens))
    print(f"Vocabulary Size: {len(vocab)}")
    
    # Initialize Term Frequency (TF) matrix
    tf = np.zeros((len(class_aggregated_tokens), len(vocab)))

    # Populate TF matrix
    for cluster_id, tokens in enumerate(class_aggregated_tokens):
        for word in vocab:
            tf[cluster_id][vocab.index(word)] = tokens.count(word)
    
    # Initialize Inverse Document Frequency (IDF) matrix
    idf = np.zeros((1, len(vocab)))
    A = tf.sum() / tf.shape[0]  # Average number of words per class
    
    # Calculate IDF for each term
    for idx, term in enumerate(vocab):
        f_t = tf[:, idx].sum()  # Frequency of term t across all classes
        idf_score = np.log(1 + (A / f_t))
        idf[0, idx] = idf_score
    
    # Calculate TF-IDF
    tf_idf = tf * idf
    
    # Extract top n terms for each class
    n = 5  # Number of top words to display
    top_idx = np.argpartition(tf_idf, -n)[:, -n:]
    top_terms_per_class = {}
    
    for cluster_id, indices in enumerate(top_idx):
        top_terms = [vocab[idx] for idx in indices]
        top_terms_per_class[cluster_id] = top_terms
    
    return top_terms_per_class, tf_idf


