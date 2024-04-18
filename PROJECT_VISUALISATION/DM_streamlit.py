'''
Author : Romy Beauté
Created on : 18/04/2024
'''

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d import Axes3D
import plotly.figure_factory as ff


from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

import umap
import hdbscan


#######################################################################################
############################ DATASET SELECTION ################################
#######################################################################################


# import helpers functions manually created
project_path = os.path.abspath('/Users/rb666/projects/TopicModelling_META')
if project_path not in sys.path:
    sys.path.append(project_path)
from helpers.get_DM_data import *
from helpers.BERT_helpers import find_topics_c_TF_IDF


# Caching functions to improve load times and performance
@st.cache(allow_output_mutation=True)
def compute_embeddings(data, model):
    sentence_model = SentenceTransformer(model)
    return sentence_model.encode(data, show_progress_bar=True)

@st.cache(allow_output_mutation=True)
def compute_umap_embeddings(embeddings, n_components, n_neighbors, min_dist):
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    return umap_model.fit_transform(embeddings)

@st.cache(allow_output_mutation=True)
def perform_hdbscan(embeddings, min_cluster_size, min_samples):
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    return hdbscan_model.fit_predict(embeddings)




def main():
    st.title('Sensory Tool Data - Topic Modelling')
    st.sidebar.title("Dataset Selection")
    add_sidebar = st.sidebar.selectbox('Select dataset', ['High Sensory', 'Deep Listening','Handwritten','Data General Infos'])

    # Load data
    HS_reports = load_DM_data(HighSensory=True,Handwritten=False)
    DL_reports = load_DM_data(HighSensory=False,Handwritten=False)
    HW_reports = load_DM_data(HighSensory=False,Handwritten=True)


    # Clean data
    HS_cleaned_reports = get_cleaned_data(HS_reports)
    DL_cleaned_reports = get_cleaned_data(DL_reports)
    HW_cleaned_reports = get_cleaned_data(HW_reports)

    hdbscan_labels = None


    #add visualisation of the data in each corresponding sidebar 
    if add_sidebar == 'High Sensory':
        reports = HS_cleaned_reports
        st.write(HS_cleaned_reports)


    elif add_sidebar == 'Deep Listening':
        st.write(DL_cleaned_reports)
        reports = DL_cleaned_reports


    elif add_sidebar == 'Handwritten':
        st.write(HW_cleaned_reports)
        reports = HW_cleaned_reports
        #Plot the number of reports before and after preprocessing for handwritten data
        initial_counts = {'Handwritten': len(HW_reports)}
        final_counts = {'Handwritten': len(HW_cleaned_reports)}

        labels = list(initial_counts.keys())
        initial_vals = [initial_counts[condition] for condition in labels]
        final_vals = [final_counts[condition] for condition in labels]

        x = range(len(labels))

        plt.figure(figsize=(10, 6))
        plt.bar(x, initial_vals, width=0.4, label='Before Preprocessing', align='center')
        plt.bar(x, final_vals, width=0.4, label='After Preprocessing', align='edge')
        plt.xlabel('Condition')
        plt.ylabel('Number of Reports')
        plt.title('Handwritten dataset : Number of Reports Before and After Preprocessing')
        plt.xticks(x, labels)
        plt.legend()
        plt.show()
        #show it in the streamlit app
        st.pyplot(plt)




    else:
        st.write('Sensory Tool Data - General Infos ')
        #Plot the number of reports before and after preprocessing
        initial_counts = {'HighSensory': len(HS_reports), 'DeepListening': len(DL_reports)}
        final_counts = {'HighSensory': len(HS_cleaned_reports), 'DeepListening': len(DL_cleaned_reports)}

        labels = list(initial_counts.keys())
        initial_vals = [initial_counts[condition] for condition in labels]
        final_vals = [final_counts[condition] for condition in labels]

        x = range(len(labels))

        plt.figure(figsize=(10, 6))
        plt.bar(x, initial_vals, width=0.4, label='Before Preprocessing', align='center')
        plt.bar(x, final_vals, width=0.4, label='After Preprocessing', align='edge')
        plt.xlabel('Condition')
        plt.ylabel('Number of Reports')
        plt.title('Sensory tool dataset : Number of Reports Before and After Preprocessing')
        plt.xticks(x, labels)
        plt.legend()
        plt.show()
        #show it in the streamlit app
        st.pyplot(plt)


    #######################################################################################
    #################### TOPIC MODELLING OPTIONS ##########################################
    #######################################################################################

    add_sidebar = st.sidebar.selectbox('Topic Modelling Parameters', ['LDA','BERTopic','BERTopic with hyperparameters tuning'])

    if add_sidebar == 'LDA':
        st.write('LDA')

        
    elif add_sidebar == 'BERTopic':
        #add different sidebars to select : the transformer model, the vectorizer model, the UMAP model and the HDBSCAN model
        add_sidebar = st.sidebar.selectbox('Select Transformer Model', ['all-MiniLM-L6-v2'])
        if add_sidebar == 'all-MiniLM-L6-v2':
            st.write('Transformer Model : all-MiniLM-L6-v2')
            model = "all-MiniLM-L6-v2"  #"paraphrase-MiniLM-L6-v2" #
            sentence_model = SentenceTransformer(model)
            embeddings = sentence_model.encode(reports.values,show_progress_bar=True)

        model = BERTopic(language="english", 
                    #  vectorizer_model=vectorizer_model, 
                    calculate_probabilities=True,
                    min_topic_size=5,
                    embedding_model=model)
        topics, probas = model.fit_transform(reports.values, embeddings=embeddings)
        # visualise the topics outputted from model.visualize_topics() in the streamlit app 
        st.write(model.visualize_topics())
        # # visualise the topics outputted from model.visualize_distribution() in the streamlit app
        # st.write(model.visualize_distribution(probabilities=True))
        # # visualise the topics outputted from model.visualize_heatmap() in the streamlit app
        # st.write(model.visualize_heatmap())
        # # visualise the topics outputted from model.visualize_barchart() in the streamlit app
        st.write(model.visualize_barchart())
        # # visualise the topics outputted from model.visualize_hierarchy() in the streamlit app
        st.write(model.visualize_hierarchy())
        # # visualise the topics outputted from model.visualize_term_rank() in the streamlit app
        st.write(model.visualize_documents(reports.values)) #Visualize the documents in each topic)


    elif add_sidebar == 'BERTopic with hyperparameters tuning':
        # Initialize the session state for HDBSCAN options 
        if 'show_hdbscan_options' not in st.session_state:
            st.session_state['show_hdbscan_options'] = False

        #add different sidebars to select : the transformer model, the vectorizer model, the UMAP model and the HDBSCAN model
        add_sidebar = st.sidebar.selectbox('Select Transformer Model', ['all-MiniLM-L6-v2'])
        if add_sidebar == 'all-MiniLM-L6-v2':
            st.write('Transformer Model : all-MiniLM-L6-v2')
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

            data = reports.tolist()  # Adjust 'cleaned_text' to your column name

            batch_size = 16 #number of doc embeddings at the time

            embeds = np.zeros((len(data), sentence_model.get_sentence_embedding_dimension()))

            for i in tqdm(range(0, len(data), batch_size)):
                i_end = min(i+batch_size, len(data))
                batch = data[i:i_end]
                batch_embed = sentence_model.encode(batch)
                embeds[i:i_end,:] = batch_embed

        # ********** PARAMS FOR UMAP **********
        # add a sidebar that allow to enter the number of components for UMAP
        n_components = st.sidebar.slider("Number of components for UMAP",2,15,5,1)

    
        # add a sidebar that allow to enter the number of neighbors for UMAP
        n_neighbors = st.sidebar.slider("Number of neighbors for UMAP",3,15,5,1)

        # add a sidebar that allow to enter the minimum distance for UMAP
        min_dist = st.sidebar.slider("Minimum distance for UMAP", 0.0, 0.1, 0.01,0.01)

        # plot the UMAP in the streamlit app with the selected parameters
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
        umap_embeddings = umap_model.fit_transform(embeds)

        if n_components == 2:
            fig = px.scatter(umap_embeddings, x=0, y=1)
            #add title to the plot and axis labels
            fig.update_layout(title='UMAP Clustering', scene = dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2'))
            st.plotly_chart(fig)
        else:
            fig = px.scatter_3d(umap_embeddings, x=0, y=1, z=2)
            # add title to the plot and axis labels
            fig.update_layout(title='UMAP Clustering', scene = dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3'))
            st.plotly_chart(fig)



        # ********** PARAMS FOR HDBSCAN **********

        #transition from UMAP to HDBSCAN selection : add a button that allow to make HDBSCAN option appear in the streamlit app
        if st.sidebar.button('HDBSCAN Options'):
            st.session_state['show_hdbscan_options'] = not st.session_state['show_hdbscan_options'] 


        # Show HDBSCAN parameters if the toggle is on
        if st.session_state['show_hdbscan_options']:
            
            # add a sidebar that allow to enter the minimum cluster size for HDBSCAN
            min_value_cs = int(len(reports)*0.01)
            max_value_cs = int(len(reports)*0.05)
            min_cluster_size = st.sidebar.slider("Minimum cluster size for HDBSCAN", min_value_cs, max_value_cs, min_value_cs,1)

            # add a sidebar that allow to enter the minimum samples for HDBSCAN
            min_samples = st.sidebar.slider("Minimum samples for HDBSCAN", min_value_cs, max_value_cs, min_value_cs,1)

            # add a sidebar that allow to enter the cluster selection method for HDBSCAN
            # cluster_selection_method = st.sidebar.selectbox("Cluster selection method for HDBSCAN", ['dbscan','eom', 'leaf', 'flat'])

            # plot the HDBSCAN in the streamlit app with the selected parameters
            hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            hdbscan_labels = hdbscan_model.fit_predict(umap_embeddings)

            plt.figure(figsize=(10, 5))
            hdbscan_model.condensed_tree_.plot(select_clusters=True,colorbar=True)
            # add title to the plot and axis labels
            plt.title('HDBSCAN Clustering')
            plt.xlabel('Samples')
            plt.ylabel('Distance')
            st.pyplot(plt)
            

        if n_components == 2:
            fig = px.scatter(umap_embeddings, x=0, y=1, color=hdbscan_labels)
            fig.update_layout(title='HDBSCAN Clustering', scene = dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2'))
            st.plotly_chart(fig)
        else:
            fig = px.scatter_3d(umap_embeddings, x=0, y=1, z=2, color=hdbscan_labels)
            # add title to the plot and axis labels
            fig.update_layout(title='HDBSCAN Clustering', scene = dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3'))
            st.plotly_chart(fig)


        # find topics with c-TF-IDF from manually defined function in BERT_helpers.py and display them in the streamlit app
        top_terms_per_class, tf_idf = find_topics_c_TF_IDF(data, hdbscan_labels)
        # Layout with expanders for better organization
        # Convert the tf_idf array to a DataFrame for better visualization
        tf_idf_df = pd.DataFrame(tf_idf)
        st.write(tf_idf_df)
        
        with st.expander("View Top Terms for Each Cluster"):
            for cluster_id, terms in top_terms_per_class.items():
                st.write(f"Cluster {cluster_id}: {', '.join(terms)}")



        # # add a sidebar that allow to enter the number of topics for BERTopic
        # nr_topics = st.sidebar.slider("Number of topics for BERTopic", 5, 15, 10,1)

        # # add a sidebar that allow to enter the top n words for BERTopic
        # top_n_words = st.sidebar.slider("Top n words for BERTopic", 5, 15, 10,1)

        # # add a sidebar that allow to enter the ngram range for BERTopic
        # n_gram_range = st.sidebar.slider("Ngram range for BERTopic", 1, 2, 1,1)




    # # Hyperparameters
    # len_HS = len(HS_cleaned_reports)
    # len_DL = len(DL_cleaned_reports)
    # HS_hyperparams = get_hyperparams(len_HS)
    # DL_hyperparams = get_hyperparams(len_DL)




if __name__ == "__main__":
    main()