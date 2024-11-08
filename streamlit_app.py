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

# Button Navigation
if st.button('About'):
    st.header("About")
    st.write("""
    Welcome to the MLBB (Mobile Legends: Bang Bang) Dashboard. This dashboard provides insights and 
    analytics on the statistics of various MLBB heroes, exploring key trends and applying machine learning 
    techniques to enhance gameplay strategies.
    """)

elif st.button('Dataset'):
    st.header("Dataset")
    st.write("Here is a preview of the dataset used in this analysis.")
    st.dataframe(df.head())


elif st.button('Value Counts'):
    st.header("Value Counts")
    column = st.selectbox("Select a column to view value counts:", df.columns)
    st.write(f"Value counts for {column}:")
    st.write(df[column].value_counts())
    

elif st.button('EDA'):
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Here, we explore the dataset through various visualizations.")


elif st.button('Machine Learning'):
    st.header("Machine Learning")
    st.write("This section applies machine learning models to the dataset.")


elif st.button('Conclusion'):
    st.header("Conclusion")
    st.write("""
    This concludes our analysis of MLBB hero statistics. The insights and model predictions here can 
    aid in better understanding hero characteristics and strategic choices in gameplay.
    """)


