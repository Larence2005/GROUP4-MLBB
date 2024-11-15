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


### Conclusion
1. Dataset Insights
Overview
- The dataset contains 114 heroes, each described by 11 attributes such as health, mana, damage, defense, and esports performance.

Descriptive Statistics
- Attributes such as HP, Phy_Damage, and Mana demonstrate significant variability, aligning with role-specific hero designs.

2. Exploratory Data Analysis
Attribute Distributions
- HP: Right-skewed distribution with a peak between 2,500–2,750, indicating a design balance around this range.
- Phy_Damage: Near-normal distribution, peaking around 120, reflecting balanced attack capabilities.
Correlation Heatmap
- Highlights moderate positive correlations, such as between HP and Phy_Defence (r = 0.31), and trade-offs like the negative correlation between Mana and Mov_Speed (r = -0.41).

Performance Insights
- Boxplots show differences in HP and Phy_Damage for winning vs. losing heroes, suggesting that higher survivability and consistent damage output positively impact success.

3. Machine Learning Classification
Random Forest Classifier
- Achieved an accuracy of 87% in predicting a hero's Primary Role based on attributes such as health, defense, damage, and mana-related stats.

Feature Importance
- Top predictors include Mana (23.87%), Phy_Defence (16.29%), HP (15.61%), and Mana_Regen (13.07%).
- Mag_Damage and Mag_Defence had negligible impact.
Role Classification
- The supervised model identified clear distinctions between roles, with balanced performance across classifications. A repeated test achieved 100% accuracy, demonstrating the model's reliability.

4. Key Takeaways
Role-Specific Traits
- Heroes in certain roles (e.g., Fighters, Mages) exhibit distinct attribute patterns, validated by the Random Forest model's feature importance rankings.

Game Balance
- The clustering of attributes like HP and Phy_Damage indicates intentional balancing of heroes to maintain fairness and variety.

Strategic Insights
- Players can leverage attribute-based patterns and machine learning insights to optimize hero selection and role allocation in both casual and competitive play.

Data-Driven Modeling
- The machine learning approach demonstrates the potential for predictive analytics in understanding and enhancing hero dynamics.

Final Thoughts
This analysis highlights the value of combining statistical, exploratory, and predictive modeling techniques to gain a comprehensive understanding of hero attributes and roles in MLBB. These findings can guide future hero design, balance adjustments, and gameplay strategies, fostering a richer gaming experience.
