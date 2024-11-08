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



#====================================DON'T CHANGE THIS====================================

# Load dataset
df = pd.read_csv("Mlbb_Heroes.csv")

# Initialize session state to store the selected page if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default to 'About' page

# Function to set the page selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar for navigation (left side)
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('MLBB Dashboard')

    # Create buttons for each page
    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'

    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Edelle Lumabi\n2. John Larence Lusaya\n3. Nick Pastiu\n4. Sophia Vitug\n 5. Daniel Santillan")






#======================DON'T CHANGE THE FUNCTIONS, JUST ADD THE CODES INSIDE THE FUNCTIONS======================

# Content based on sidebar selection
if st.session_state.page_selection == 'about':
    st.header("About")
    st.write("""
    Welcome to the MLBB (Mobile Legends: Bang Bang) Dashboard. This dashboard provides insights and 
    analytics on the statistics of various MLBB heroes, exploring key trends and applying machine learning 
    techniques to enhance gameplay strategies.
    """)
    
    image_path = "MLBB.jpg"
    st.image(image_path, use_column_width=True)



elif st.session_state.page_selection == 'dataset':
    st.header("Dataset")
    st.write("Here is a preview of the dataset used in this analysis.")
    st.write(df)
    print(df.info())
    df.describe()



elif st.session_state.page_selection == 'eda':
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Here, we explore the dataset through various visualizations.")



elif st.session_state.page_selection == 'data_cleaning':
    st.header("Data Cleaning / Pre-processing")
    st.write("This section covers the data cleaning and pre-processing steps.")



elif st.session_state.page_selection == 'machine_learning':
    st.header("Machine Learning")
    st.write("This section applies machine learning models to the dataset.")



elif st.session_state.page_selection == 'prediction':
    st.header("Prediction")
    st.write("This section provides the prediction models and their results.")



elif st.session_state.page_selection == 'conclusion':
    st.header("Conclusion")
    st.write("This section concludes the analysis with key findings.")



