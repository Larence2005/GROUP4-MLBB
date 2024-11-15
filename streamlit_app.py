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
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
        
    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)):
            st.session_state.page_selection = 'prediction'
        
    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    st.subheader("Members")
    st.markdown("1. Edelle Lumabi\n2. John Larence Lusaya\n3. Nick Pastiu\n4. Sophia Vitug\n 5. Daniel Santillan")






#======================DON'T CHANGE THE ITERATION STATEMENTS, JUST ADD THE CODES INSIDE THE LOOPS======================

# Content based on sidebar selection

#ABOUT
if st.session_state.page_selection == 'about':
    st.title("ℹ️ About")
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
    st.title("📊 Dataset")
    st.write("Here is a preview of the dataset used in this analysis. The Dataset contains the Stats of heroes until Mobile Legends Version Patch 1.7.20 September 20, 2022.")
    st.markdown("""
    
    **Content**  
    The dataset has **114** rows containing **11 primary attributes** that are related to MLBB heroes, the columns are as follows: 
    1. Name\n
    2. Title\n
    3. Voice_Line\n
    4. Release_Date\n
    5. Primary_Role\n
    6. Secondary_Role\n
    7. Lane\n
    8. Hp\n
    9. Hp_Regen\n
    10. Mana\n
    11. Mana_Regen\n
    12. Phy_Damage\n
    13. Mag_Damage\n
    14. Phy_Defence\n
    15. Mag_Defence\n
    16. Mov_Speed\n
    17. Esport_Wins\n
    18. Esport_Loss\n

    `Link:` https://www.kaggle.com/datasets/kishan9044/mobile-legends-bang-bang
    
    """)
    
    st.subheader("Dataset displayed as a Data Frame")
    st.write(df)
    
    st.subheader("Descriptive Statistics")
    describe = df.describe()
    describe

    st.markdown("""

    The results from `df.describe()` highlights the descriptive statistics about the dataset.

    1. Hp has an average value of 2577.85 with a standard deviation of 213.28, indicating moderate variation in health points across the data. The values range from 918 to 2909, showing a significant spread in HP values.

    2. Hp_Regen has a mean of 7.70 and a standard deviation of 1.64, suggesting relatively consistent health regeneration rates, with values ranging from 3.8 to 18.4.

    3. Mana has a mean of 339.37 with a standard deviation of 215.07, reflecting a high level of variability. The range of values is from 0 (possibly indicating characters with no mana) to 750.

    4. Mana_Regen has a mean of 14.80 and a standard deviation of 22.92, indicating substantial variation in mana regeneration. The minimum value is 0, and the maximum reaches 240.
    
    5. Phy_Damage has an average of 115.93 and a smaller standard deviation of 9.83, suggesting that physical damage values are relatively stable, ranging from 90 to 140.
    
    6. Phy_Defence has a mean of 19.20 and a standard deviation of 3.58, indicating moderate consistency. Values range from 10 to 27.
    
    7. Mag_Defence has a mean of 14.91 and a low standard deviation of 0.66, suggesting high consistency across entries, with values between 10 and 15.
    
    8. Mov_Speed has an average of 251.18 with a standard deviation of 8.67, showing a small spread around the mean. The movement speed values range from 240 to 270.
    
    9. Esport_Wins has a mean of 289.86 and a high standard deviation of 275.36, indicating a wide range of values, from 0 to 1357.
    
    10. Esport_Loss similarly has a mean of 289.71 and a high standard deviation of 273.14, with values from 3 to 1529, suggesting large variability in losses.
    
    In terms of percentiles (25th, 50th, and 75th), the gradual increase across features like Hp, Mana, Phy_Damage, and Mov_Speed suggests a dataset with a broad distribution, offering potential for classification and analysis techniques to explore patterns based on these attributes.
                    
    """)

#EDA

elif st.session_state.page_selection == 'eda':
    st.title("📈 Exploratory Data Analysis (EDA)")
    st.write("Here, we explore the dataset through various visualizations.")

    with st.expander('Legend', expanded=True):
            st.write('''
                - Data: [MLBB Dataset](https://www.kaggle.com/datasets/kishan9044/mobile-legends-bang-bang).
                - :orange[**Pie Chart**]: Distribution of the Primary and Secondary roles in the dataset.
                - :orange[**Histograms**]: Distribution of HP and Physical Damage of heroes.
                - :orange[**Correlation Heatmap**]: Correlations between different numerical variables features positive correlations as ranges of red and negative correlations as shades of blue.
                - :orange[**Boxplot**]: Distribution of HP and Physical Damage of heroes by Win/Loss status.
                ''')

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
    
    # EDA for Primary Role
    st.header("Summary Statistics for Primary Roles")
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
    
    
    # EDA for Secondary Role
    st.header("Summary Statistics for Secondary Roles")
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
    st.header("Distribution of Hp")
    
    # Histogram with KDE
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
  
    # Title
    st.header("Physical Damage Analysis")
    
    # Create histogram with KDE
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Phy_Damage'], kde=True, bins=10, color='green', ax=ax)
    ax.set_title('Distribution of Physical Damage')
    ax.set_xlabel('Physical Damage')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.markdown("""
The graph shows physical damage distribution, with a histogram and a superimposed kernel density estimate (green line). The data is roughly normal, peaked around 120, and tapers off to both smaller and higher damage values. Most of the observations fall in the range of 110-130, with fewer cases below 100 and even fewer above 130.
    """)



    #-----------------CORRRELATION HEATMAP

    st.header("Correlation Heatmap for Numerical Variables")

    correlation = df[['Hp', 'Mana', 'Phy_Damage', 'Mag_Damage', 'Phy_Defence', 'Mag_Defence', 'Mov_Speed']].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap for Numerical Variables')
    st.pyplot(fig)
    st.markdown("""
The heatmap of correlations between different numerical variables features positive correlations as ranges of red and negative correlations as shades of blue. The most important observations include "Physical Defence" and "Movement Speed" with a correlation of 0.42, "Mana" and "Movement Speed" with -0.41, and "HP" with "Physical Defence" correlated with 0.31. Almost all the remaining connections are weak. Thus, a minimal number of linear dependencies between those variable pairs appear to exist.
    """)



    #----------BOXPLOT HP WIN/LOSS-----------
   
 # Create Win column based on Esport wins vs losses
    df['Win'] = df['Esport_Wins'] > df['Esport_Loss']
    
    # Create boxplot
    st.header("HP Distribution for Winning vs Losing Heroes")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Win', y='Hp', data=df, palette='coolwarm', ax=ax)
    ax.set_title('Hp vs Win/Loss')
    ax.set_xlabel('Win (True) / Loss (False)')
    ax.set_ylabel('Hp')
    
    # Display the plot
    st.pyplot(fig)
    
    # Add explanation
    st.write("""
    This boxplot shows the distribution of HP values for heroes based on their win/loss record:
    - True indicates heroes with more wins than losses
    - False indicates heroes with more losses than wins
    
    The box shows the quartiles of the distribution while the whiskers extend to show the rest of the distribution.
    """)

    #-----------BOXPLOT PHYS DMG WIN/LOSS--------
   # Create Win column based on Esport wins vs losses
    df['Win'] = df['Esport_Wins'] > df['Esport_Loss']

    
    # Create boxplot
    st.header("Physical Damage Distribution for Winning vs Losing Heroes")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Win', y='Phy_Damage', data=df, palette='viridis', ax=ax)
    ax.set_title('Physical Damage vs Win/Loss')
    ax.set_xlabel('Win (True) / Loss (False)')
    ax.set_ylabel('Physical Damage')
    
    # Display the plot
    st.pyplot(fig)
    
    # Add explanation
    st.write("""
    This boxplot shows the distribution of Physical Damage values for heroes based on their win/loss record:
    - True indicates heroes with more wins than losses
    - False indicates heroes with more losses than wins
    
    The box represents the interquartile range (IQR), with the middle line showing the median value.
    """)


#DATA_CLEANING

elif st.session_state.page_selection == 'data_cleaning':
    st.title("🧼 Data Cleaning / Pre-processing")
    st.write("This section covers the data cleaning and pre-processing steps.")
    #-------------------------------------
    st.header("Fixing null values")
    st.subheader("Checking for missing values")
    missing_count = df.isnull().sum()
    st.write(missing_count)
    #-------------------------------------
    st.subheader("Replacing null values in Secondary_Role")
    st.code("""
    df['Secondary_Role'].fillna('No Secondary Role', inplace=True)
    """)
    df['Secondary_Role'].fillna('No Secondary Role', inplace=True)
    st.subheader("Checking for missing values after cleaning")
    missing_count = df.isnull().sum()
    st.write(missing_count)
    #-------------------------------------
    st.header("Filter out rows where 'Secondary_Role' is 'No Secondary Role'")
    st.code("""
    mask = df['Secondary_Role'] != 'No Secondary Role'
    X_filtered = df[selected_features][mask]
    y_filtered = df['Secondary_Role'][mask]
    """)
    #-------------------------------------
    st.header("Label Encoding Target Variable")
    st.code("""
    le = LabelEncoder()
    y_filtered_encoded = le.fit_transform(y_filtered)
    """)
    #-----------------------------------------------
    st.header("Resampling the Data")
    st.code("""
    X_resampled, y_resampled = resample(X_filtered, y_filtered_encoded,
                                   n_samples=resample_size,
                                   random_state=42)
    """)
    #-----------------------------------------------
    st.header("Scaling the features")
    st.code("""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    """)
    #-----------------------------------------------
    st.header("Computing class weights")
    st.code("""
    class_weight_dict = dict(zip(np.unique(y_resampled),
                             compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_resampled),
                                                  y=y_resampled)))
    """)
    st.write("After cleaning the data, it is now ready for the machine learning model, addressing data quality issues and ensuring the model can learn effectively from the available information.")


# MACHINE LEARNING
elif st.session_state.page_selection == 'machine_learning':
    st.header("Machine Learning")
    st.write("This section applies machine learning models to the dataset.")
    
    st.title("Primary Roles of MLBB Heroes Classification using Random Forest Model")
    st.write("This uses a Random Forest model to classify MLBB heroes primary roles based on selected features.")
    
    # Feature selection
    selected_features = [
        'Hp', 'Hp_Regen', 'Mana', 'Mana_Regen',
        'Phy_Damage', 'Mag_Damage', 'Phy_Defence', 'Mag_Defence',
        'Mov_Speed', 'Esport_Wins', 'Esport_Loss'
    ]
    
    # Data preparation
    X = df[selected_features]
    y = df['Primary_Role']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Model creation and training
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Display model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.3f}")
    
    # Generate classification report with output_dict=True
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    # Display the classification report as a table in Streamlit
    st.subheader("Classification Report")
    st.dataframe(report_df)
    
    # Feature importance visualization
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    # Plot feature importance
    st.subheader("Feature Importance in Predicting MLBB Hero Primary Role")
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'])
    
    # Add labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width*100:.1f}%', ha='left', va='center')
    
    ax.set_xlabel('Feature Importance (%)')
    ax.set_title('Feature Importance in Predicting MLBB Hero Primary Role')
    st.pyplot(fig)
    
    # Display feature importance percentages
    st.subheader("Feature Importance Percentages")
    for feature, importance in zip(feature_importance['Feature'], feature_importance['Importance']):
        st.write(f"{feature}: {importance * 100:.2f}%")
    
    
    st.write("\n")
    st.write("On the other hand, the data below uses Random Forest model to classify the secondary roles of heroes based on some hypothetical features and predict the secondary role of the hero based on the input features.")
    
    st.title("Secondary Roles of MLBB Heroes Prediction Using Random Forest Model")
    
    # Define role data for display
    data = {
        'Secondary_Role': ['No Secondary Role', 'Support', 'Tank', 'Assassin', 'Mage', 'Fighter', 'Marksman'],
        'Count': [84, 7, 6, 6, 5, 3, 3],
    }
    
    # Display role distribution
    st.subheader("Secondary Role Distribution")
    role_distribution_df = pd.DataFrame(data)
    role_distribution_df['Percentage'] = (role_distribution_df['Count'] / role_distribution_df['Count'].sum()) * 100
    st.dataframe(role_distribution_df)
    
    # Define color scheme
    role_colors = {
        'No Secondary Role': '#808080',
        'Support': '#4FB9E3',
        'Tank': '#2ECC71',
        'Assassin': '#E74C3C',
        'Mage': '#9B59B6',
        'Fighter': '#E67E22',
        'Marksman': '#F1C40F'
    }
    
    # Define features and labels
    selected_features = [
        'Hp', 'Hp_Regen', 'Mana', 'Mana_Regen',
        'Phy_Damage', 'Mag_Damage', 'Phy_Defence', 'Mag_Defence',
        'Mov_Speed', 'Esport_Wins', 'Esport_Loss'
    ]
    
    X = df[selected_features]
    y = df['Secondary_Role']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    
    # Filter out "No Secondary Role"
    mask = y != 'No Secondary Role'
    X_filtered = X_scaled[mask]
    y_filtered = y[mask]
    
    # Encode target variable
    le = LabelEncoder()
    y_filtered_encoded = le.fit_transform(y_filtered)
    
    # Resample data
    X_resampled, y_resampled = resample(X_filtered, y_filtered_encoded,
                                        n_samples=30,  # Removed the slider part
                                        random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                        test_size=0.2,  # Removed the slider part
                                                        random_state=42)
    
    # Compute class weights
    class_weight_dict = dict(zip(np.unique(y_train),
                                 compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train)))
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight=class_weight_dict
    )
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.3f}")
    
    # Display classification report
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [le.classes_[i] for i in unique_classes]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.subheader("Classification Report")
    st.write(pd.DataFrame(report).transpose())
    
    # Visualize feature importance
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=True)
    
    # Plotting feature importances
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=[role_colors['Support'], role_colors['Tank'], role_colors['Assassin'], role_colors['Mage'], role_colors['Fighter'], role_colors['Marksman']])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance in Predicting Secondary Roles")
    for i, (bar, importance) in enumerate(zip(bars, importance_df['Importance'])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2,
                f'{importance * 100:.2f}%', ha='left', va='center', fontweight='bold')
    st.pyplot(fig)
    
    # Display feature importance as text
    st.subheader("Feature Importance Percentages")
    for feature, importance in zip(importance_df['Feature'], importance_df['Importance']):
        st.write(f"{feature}: {importance * 100:.2f}%")
    
    
    # SUPERVISED LEARNING
    st.title("Supervised Learning for Secondary Role Distribution")
    
    # Define role data
    data = {
        'Secondary_Role': ['Support', 'Tank', 'Assassin', 'Mage', 'Fighter', 'Marksman'],
        'Count': [7, 6, 6, 5, 3, 3]
    }
    
    # Display role distribution as a table
    st.subheader("Role Distribution")
    df = pd.DataFrame(data)
    st.table(df)
    
    # Repeat roles based on count
    roles = []
    for role, count in zip(df['Secondary_Role'], df['Count']):
        roles.extend([role] * count)
    
    df_repeated = pd.DataFrame({'Secondary_Role': roles})
    
    # Display repeated roles count
    st.subheader("Repeated Roles Count")
    st.write(df_repeated['Secondary_Role'].value_counts())
    
    # Feature extraction with TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_repeated['Secondary_Role'])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_repeated['Secondary_Role'])
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Naive Bayes model
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    
    # Make predictions and evaluate
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Ensure all classes are included in classification report
    target_names = label_encoder.classes_
    
    # Get unique classes from both actual and predicted values
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    
    # Generate classification report with correct labels
    report = classification_report(y_test, y_pred, target_names=target_names, labels=unique_classes, output_dict=True)
    
    # Convert classification report into a pandas DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    # Display accuracy and classification report as a table
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.3f}")
    st.subheader("Classification Report")
    st.dataframe(report_df)

#PREDICTION
X = df[selected_features]
y_primary = df['Primary_Role']  # Adjusted for primary role
y_secondary = df['Secondary_Role']  # Adjusted for secondary role

# Define scaler and scale your features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=selected_features)

# Split data and define models here
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_primary, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
model.fit(X_train, y_train)

# Define secondary role classifier (Decision Tree as an example)
dt_classifier = DecisionTreeClassifier(random_state=42)
y_secondary_encoded = LabelEncoder().fit_transform(y_secondary)
X_train, X_test, y_train_secondary, y_test_secondary = train_test_split(X_scaled, y_secondary_encoded, test_size=0.2, random_state=42)
dt_classifier.fit(X_train, y_train_secondary)

# Save list of classes for secondary role predictions
classes_list = LabelEncoder().fit(y_secondary).classes_

# Prediction section
elif st.session_state.page_selection == 'prediction':
    st.title("Prediction")

    # Define input data
    input_data = {
        'Hp': 5000,
        'Hp_Regen': 50,
        'Mana': 2000,
        'Mana_Regen': 20,
        'Mag_Damage': 0,
        'Mag_Defence': 0,
        'Phy_Damage': 150,
        'Phy_Defence': 30,
        'Mov_Speed': 270,
        'Esport_Wins': 400,
        'Esport_Loss': 350
    }

    # Ensure input data order matches the training features
    input_data_scaled = scaler.transform([list(input_data.values())])
    input_data_scaled = pd.DataFrame(input_data_scaled, columns=selected_features)

    # Make predictions
    primary_role_prediction = model.predict(input_data_scaled)[0]
    secondary_role_prediction = dt_classifier.predict(input_data_scaled)[0]

    # Display predictions
    st.write(f"Predicted Primary Role: {primary_role_prediction}")
    st.write(f"Predicted Secondary Role: {classes_list[secondary_role_prediction]}")
    
    

#CONCLUSION
elif st.session_state.page_selection == 'conclusion':
    st.title("📝 Conclusion")
    st.write("This section concludes the analysis with key findings.")
