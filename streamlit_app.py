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

if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'

def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:

    st.title('MLBB Dashboard')

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

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    st.subheader("Members")
    st.markdown("1. Edelle Lumabi\n2. John Larence Lusaya\n3. Nick Pastiu\n4. Sophia Vitug\n 5. Daniel Santillan")






#======================DON'T CHANGE THE ITERATION STATEMENTS, JUST ADD THE CODES INSIDE THE LOOPS======================

# Content based on sidebar selection

#ABOUT
if st.session_state.page_selection == 'about':
    st.header("About")
    st.write("""
    Welcome to the MLBB (Mobile Legends: Bang Bang) Dashboard. This dashboard provides insights and 
    analytics on the statistics of various MLBB heroes, exploring key trends and applying machine learning 
    techniques to enhance gameplay strategies.
    """)
    
    image_path = "MLBB.jpg"
    st.image(image_path, use_column_width=True)

    st.markdown("""

    #### Pages
    1. `Dataset` - Brief description of the Mobile Legends: Bang Bang (MLBB) dataset used in this dashboard. 
    2. `EDA` - Exploratory Data Analysis of MLBB dataset. Highlighting the distribution of Primary and Secondary Roles, HP, and Physical Damage. It also highlights the frequency of primary roles. Includes graphs such as Pie Chart, Histograms, Bar Graphs, Heatmaps, and Boxplots
    3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as replacing null values in the Secondary Roles column into strings.
    4. `Machine Learning` - Training two supervised classification models: Supervised Learning and Random Forest Regressor.
    5. `Prediction` - Prediction page where users can input values to predict the the primary role of the hero based on the input features using the trained models.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.


    """)
#DATASET

elif st.session_state.page_selection == 'dataset':
    st.header("📊 Dataset")
    st.write("Here is a preview of the dataset used in this analysis. The Dataset contains the Stats of heroes until Mobile Legends Version Patch 1.7.20 September 20, 2022.")
    st.markdown("""**Content**  
    The dataset has **114** rows containing **_ primary attributes** that are related to MLBB heroes, the columns are as follows: .

    `Link:` https://www.kaggle.com/datasets/kishan9044/mobile-legends-bang-bang         
    """)
    st.write(df)
    describe = df.describe()
    describe

    st.markdown("""

    The results from `df.describe()` highlights the descriptive statistics about the dataset. First the **sepal length** averages *5.84 cm* with a standard deviation of *0.83* which indicates moderate variation around the mean. **Sepal width** on the other hand has a lower mean of *3.05* cm and shows less spread with a standard deviation of *0.43*, this indicates that the values of sepal width are generally more consistent. Moving on with **petal length** and **petal width**, these columns show greater variability with means of *3.76 cm* and *1.20 cm* and standard deviation of *1.76* and *0.76*. This suggests that these dimansions vary more significantly across the species.  

    Speaking of minimum and maximum values, petal length ranges from *1.0 cm* up to *6.9 cm*, petal width from *0.1 cm* to *2.5 cm* suggesting that there's a distinct difference between the species.  

    The 25th, 50th, and 75th percentiles on the other hand reveals a gradual increase across all features indicating that the dataset offers a promising potential to be used for classification techniques.
                
    """)

#EDA

elif st.session_state.page_selection == 'eda':
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Here, we explore the dataset through various visualizations.")

    #---------PRIMARY AND SECONDARY ROLE--------
    
    # Data for Primary Role
    primary_data = {
        'Primary_Role': ['Fighter', 'Mage', 'Marksman', 'Tank', 'Assassin', 'Support'],
        'Count': [33, 25, 18, 16, 13, 9]
    }
    primary_df = pd.DataFrame(primary_data)
    
    # Data for Secondary Role
    secondary_data = {
        'Secondary_Role': ['No Second Role', 'Support', 'Tank', 'Assassin', 'Mage', 'Fighter', 'Marksman'],
        'Count': [84, 7, 6, 6, 5, 3, 3]
    }
    secondary_df = pd.DataFrame(secondary_data)
    
    # Title
    st.title("MLBB Heroes Role Analysis")
    
    # Primary Role Analysis
    st.header("Primary Role Analysis")
    
    # EDA for Primary Role
    st.subheader("Summary Statistics for Primary Roles")
    st.write(primary_df.describe())
    
    # Total heroes for Primary Role
    total_primary_heroes = primary_df['Count'].sum()
    st.write(f"**Total number of heroes:** {total_primary_heroes}")
    
    # Frequency distribution for Primary Role
    st.subheader("Frequency Distribution by Primary Role")
    st.write(primary_df)
    
    # Pie chart for Primary Role
    st.subheader("Distribution of Primary Roles of Heroes in MLBB")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.pie(primary_df['Count'], labels=primary_df['Primary_Role'], autopct='%1.1f%%',
            colors=['blue', 'green', 'red', 'purple', 'orange', 'pink'], startangle=90)
    ax1.set_title("Distribution of Primary Roles of Heroes in MLBB")
    ax1.axis('equal')
    st.pyplot(fig1)

    st.markdown("""

    As displayed in this exploratory data analysis, it reveals that the **Fighter** role has the highest count with 33 heroes, 
    while **Support** has the least with 9 heroes, out of the total of 114 heroes. The summary statistics show a mean of **19 heroes per  role**, 
    with **a standard deviation of 8.69**, indicating moderate variability in the distrubution of heroes across each role. In addition, the pie chart provided visualizes the 
    **proportional distrubution of the heroes** of **Mobile Legends: Bang Bang** based on the dataset chosen for this project.
                
    """)
    
    # Secondary Role Analysis
    st.header("Secondary Role Analysis")
    
    # EDA for Secondary Role
    st.subheader("Summary Statistics for Secondary Roles")
    st.write(secondary_df.describe())
    
    # Total heroes for Secondary Role
    total_secondary_heroes = secondary_df['Count'].sum()
    st.write(f"**Total number of heroes with secondary roles:** {total_secondary_heroes}")
    
    # Frequency distribution for Secondary Role
    st.subheader("Frequency Distribution by Secondary Role")
    st.write(secondary_df)
    
    # Pie chart for Secondary Role
    st.subheader("Distribution of Secondary Roles of Heroes in MLBB")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.pie(secondary_df['Count'], labels=secondary_df['Secondary_Role'], autopct='%1.1f%%',
            colors=['pink', 'purple', 'orange', 'green', 'blue', 'red'], startangle=90)
    ax2.set_title("Distribution of Secondary Roles of Heroes in MLBB")
    ax2.axis('equal')
    st.pyplot(fig2)

    st.markdown("""

   Based on this exploratory data analysis, it illustrates the distrubution of heroes based on their secondary roles in 
   **Mobile Legends: Bang Bang (MLBB**). It gives vital statistical results such as the mean and standard deviation of the
   number of heroes per role, revealing that **"Support"** has the most (7), while **"Fighter"** and **"Marksmen"** have the fewest 
   (3 each). The overall number of heroes throughout all secondary roles is 30, 
   and the data is represented by a pie chart, which helps in illustrating the proportionate distribution of each role.
   
    """)

    #---------DISTRIBUTION OF HP-----------
    
    df = pd.read_csv("Mlbb_Heroes.csv")
    
    # Title
    st.title("Distribution of Hp")
    
    # Histogram with KDE
    st.subheader("Hp Distribution Histogram")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Hp'], kde=True, bins=10, color='blue', ax=ax)
    ax.set_title('Distribution of Hp')
    ax.set_xlabel('Hp')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.markdown("""

    This histogram plots the distribution of HP values with a superimposed kernel density estimate (the blue line) to give 
    a representation of the data's frequency. The distribution is right-skewed, meaning it has a longer tail on the higher end of the 
    HP scale. Note most values cluster between 2,250 and 3,000. It shows an apparent peak around 2,500–2,750. Characters that have very low HP, 
    about 1,000–1,250 are relatively few. Thus, there seems to be a 
    pattern where the game or system design does keep most characters' HP in some kind of "sweet spot" near 2,500, while higher or lower values seem less common.
                
    """)

    #-----------PHYSICAL DAMAGE ANALYSIS-----------
  

    #-------------PRIMARY ROLE DISTRIBUTION---------
    
    #----------CORRELATION HEATMAP-------------


    #----------BOXPLOT HP WIN/LOSS-----------
   

    #-----------BOXPLOT PHYS DMG WIN/LOSS--------
   

#DATA_CLEANING

elif st.session_state.page_selection == 'data_cleaning':
    st.header("Data Cleaning / Pre-processing")
    st.write("This section covers the data cleaning and pre-processing steps.")

#MACHINE LEARNING

elif st.session_state.page_selection == 'machine_learning':
    st.header("Machine Learning")
    st.write("This section applies machine learning models to the dataset.")

#CONCLUSION

elif st.session_state.page_selection == 'conclusion':
    st.header("Conclusion")
    st.write("This section concludes the analysis with key findings.")



