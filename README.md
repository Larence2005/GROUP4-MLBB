# GROUP-4-MLBB
A Streamlit web application that performs Exploratory Data Analysis (EDA), Data Preprocessing, and Supervised Machine Learning to classify Primary Roles of MLBB heroes using Decision Random Forest Regressor.

![image](https://github.com/user-attachments/assets/f34fe39e-8a20-400d-8417-28a0fbd1cedb)

### 👨‍👩‍👧‍👦 Members:
1. Edelle Lumabi
2. John Larence Lusaya
3. Nick Pastiu
4. Daniel Santillan
5. Sophia Vitug

### 🔗 Links:

- 🌐 [Streamlit Link](https://group4-mlbb.streamlit.app/)
- 📗 [Google Colab Notebook](https://colab.research.google.com/drive/17DEuy7XGnMU2hIcqzZBo11P3-W6-fxUE#scrollTo=kqT3DaYpC20d)

### 📊 Dataset:

- [Mobile Legends: Bang Bang E-sports Heroes Stats (Kaggle)](https://www.kaggle.com/datasets/kishan9044/mobile-legends-bang-bang)

### 📖 Pages:
- Dataset - Brief description of the Mobile Legends: Bang Bang (MLBB) dataset used in this dashboard.
- EDA - Exploratory Data Analysis of MLBB dataset. Highlighting the distribution of Primary and Secondary Roles, HP, and Physical Damage. It also highlights the frequency of primary roles. Includes graphs such as - Pie Chart, Histograms, Bar Graphs, Heatmaps, and Boxplots
- Data Cleaning / Pre-processing - Data cleaning and pre-processing steps such as replacing null values in the Secondary Roles column into strings.
- Machine Learning - Training two supervised classification models: Supervised Learning and Random Forest Regressor.
- Prediction - Prediction page where users can input values to predict the the primary role of the hero based on the input features using the trained models.
- Conclusion - Summary of the insights and observations from the EDA and model training.

### 💡 Findings / Insights

Through exploratory data analysis and training of one classification models (`Random Forest Regressor`) on the **Mobile Legends: Bang Bang E-sports Heroes Dataset**, the key insights and observations are:

#### 1. 📊 **Dataset Characteristics**:

- 

#### 2. 📝 **Feature Distributions and Separability**:

- 

#### 3. 📈 **Model Performance (Decision Tree Classifier)**:

- The `Decision Tree Classifier` achieved 100% accuracy on the training data which suggests that using a relatively simple and structured dataset resulted in a strong performance for this model. However, this could also imply potential **overfitting** due to the model's high sensitivity to the specific training samples.
- In terms of **feature importance** results from the _Decision Tree Model_, `petal_length` was the dominant predictor having **89%** importance value which is then followed by `petal_width` with **8.7%**.

##### **Summing up:**

Throughout this data science activity, it is evident that 
