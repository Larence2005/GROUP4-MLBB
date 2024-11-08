import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import altair as alt
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer



# Load dataset
df = pd.read_csv("Mlbb_Heroes.csv")

# Title
st.title("MLBB Dashboard")

# Initialize session state to store the selected option
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None

# Sidebar for navigation (left side)
with st.sidebar:
    st.header('Navigation')
    sidebar_options = ['About', 'Dataset', 'Value Counts', 'EDA', 'Machine Learning', 'Conclusion', 'Members']
    
    # Create buttons for each option
    for option in sidebar_options:
        if st.button(option):
            st.session_state.selected_option = option

# Content based on sidebar selection
selected_option = st.session_state.selected_option

if selected_option == 'About':
    st.header("About")
    st.write("""
    Welcome to the MLBB (Mobile Legends: Bang Bang) Dashboard. This dashboard provides insights and 
    analytics on the statistics of various MLBB heroes, exploring key trends and applying machine learning 
    techniques to enhance gameplay strategies.
    """)

elif selected_option == 'Dataset':
    st.header("Dataset")
    st.write("Here is a preview of the dataset used in this analysis.")
    st.write(df)

elif selected_option == 'Value Counts':
    st.header("Value Counts")
    column = st.selectbox("Select a column to view value counts:", df.columns)
    st.write(f"Value counts for {column}:")
    st.write(df[column].value_counts())

elif selected_option == 'EDA':
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Here, we explore the dataset through various visualizations.")

elif selected_option == 'Machine Learning':
    st.header("Machine Learning")
    st.write("This section applies machine learning models to the dataset.")

elif selected_option == 'Conclusion':
    st.header("Conclusion")
    st.write("This section concludes the analysis with key findings.")
