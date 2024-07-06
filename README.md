### Advanced Text Classification and Wine Quality Classification

---

## Introduction

### Project Overview

In today’s data-centric world, understanding and analyzing textual and numerical data is essential for informed decision-making. This comprehensive report delves into two distinct but related projects: advanced text classification using machine learning techniques and the classification of wine quality through hyperparameter tuning. The core aim of the first project is to transform raw textual data into actionable insights by classifying it into predefined categories, demonstrating the application of natural language processing (NLP) and classification algorithms to real-world challenges like spam detection and sentiment analysis. The second project focuses on improving the accuracy of wine quality prediction by refining a Support Vector Classifier (SVC) through hyperparameter optimization, supporting quality control in the wine industry by predicting wine quality based on chemical properties.

### Personal Motivation

My journey into data science is driven by a deep-seated passion for discovering patterns in diverse data types. With a background in biochemistry and bioinformatics, I developed an early appreciation for the complexities of data analysis. The choice to explore advanced text classification stemmed from my aspiration to apply machine learning to ubiquitous and intricate text data, aligning with my career goal of becoming proficient in NLP. Simultaneously, the wine quality classification project provided an exciting opportunity to apply model optimization techniques to a practical problem, deepening my understanding of hyperparameter tuning while contributing to real-world industry applications.

---

## Methodology

### Advanced Text Classification

#### Data Collection and Preparation

**Dataset**: The text dataset was provided by ExploreAI Academy, consisting of various text samples categorized into relevant classes. The data collection involved extracting textual data from a comprehensive repository and splitting it into training and testing subsets.

**Challenges Faced**:

- **Handling Imbalanced Data**: Some classes were underrepresented, potentially biasing the model.
- **Preprocessing Complexity**: Diverse text formats required detailed cleaning.

**Data Cleaning and Preprocessing**:

- **Handling Missing Values**: Removed or imputed missing data.
- **Text Normalization**: Lowercased text, removed punctuation, and applied stemming or lemmatization.
- **Feature Engineering**: Used techniques like CountVectorizer and TF-IDF to convert text into numerical vectors.

#### Exploratory Data Analysis (EDA)

**Insights**:

- **Word Frequency**: Visualized commonly occurring words using word clouds.
- **Class Distribution**: Showed class distribution using bar charts, highlighting imbalances.
- **Bivariate Analysis**: Explored the relationship between text length and class labels.

**Visualizations**:

- **Word Clouds**: Illustrated word frequencies in each category.
- **Class Distribution Plots**: Displayed the number of samples per class.
- **Text Length Histograms**: Provided insights into the typical text lengths.

### Wine Quality Classification

#### Data Collection and Preparation

**Data Sources**: The Wine Quality dataset from the UCI Machine Learning Repository, including chemical properties and quality scores for red and white Portuguese "Vinho Verde" wine.

**Data Collection**: Downloaded from the UCI repository as a CSV file, featuring attributes like acidity, sugar content, chlorides, etc.

**Data Cleaning and Preprocessing**:

- **Missing Values**: Filled missing values with zeros to maintain dataset integrity.
- **Feature Scaling**: Standardized features using StandardScaler.
- **Label Binarization**: Converted quality scores into binary labels (low quality ≤ 4, high quality ≥ 5).

```python
def data_preprocess(df):
    df['quality'] = df['quality'].apply(lambda x: 0 if x <= 4 else 1)
    df.fillna(0, inplace=True)
    X = df.drop('quality', axis=1)
    y = df['quality']
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    return (X_train, y_train), (X_test, y_test)
```

#### Exploratory Data Analysis (EDA)

**Insights**:

- **Feature Correlations**: Alcohol content and sulphates showed significant patterns related to wine quality.
- **Guided Feature Selection**: Insights guided the selection and preprocessing steps for model training.

---

## Modeling and Implementation

### Advanced Text Classification

#### Model Selection

**Models Considered**:

- **Logistic Regression**: Chosen for simplicity in binary classification.
- **Random Forest**: Considered for handling non-linear relationships.
- **Support Vector Machine (SVM)**: Known for high performance in text classification.
- **Multi-layer Perceptron (MLP)**: Evaluated for neural network capabilities.

**Final Model(s) Chosen**:

- **Logistic Regression**: Selected for interpretability.
- **SVM with RBF Kernel**: Chosen for its superior performance in high-dimensional spaces.

**Training Process**:

- **Hyperparameter Tuning**: Used Grid Search with cross-validation for optimal parameters.
- **Validation**: Implemented k-fold cross-validation.

```python
# Logistic Regression Implementation
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)

# SVM Implementation
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)
```

### Wine Quality Classification

#### Model Selection

**Considered Models**:

- **Logistic Regression**
- **Decision Trees**
- **Support Vector Classifier (SVC)**

**Final Model Chosen**: **SVC**, for its robustness in handling high-dimensional data and non-linear relationships.

**Model Training**:

- **Initial Training**: Trained with default parameters.
- **Hyperparameter Tuning**: Conducted grid search for optimal SVC parameters.

```python
def train_SVC_model(X_train, y_train):
    svc_model = SVC(random_state=40, gamma='auto')
    svc_model.fit(X_train, y_train)
    return svc_model
svc = train_SVC_model(X_train, y_train)
```

#### Implementation Details

**Tools Used**:

- **Python Libraries**: Scikit-Learn, Pandas, NumPy for data manipulation, model building, and evaluation.
- **Custom Scoring Function**: Evaluated model performance through log-loss scoring.

```python
def custom_scoring_function(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.maximum(epsilon, y_pred)
    y_pred = np.minimum(1 - epsilon, y_pred)
    log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return round(log_loss, 7)

def tune_SVC_model(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
    svc = SVC(random_state=40, gamma='auto')
    custom_scorer = make_scorer(custom_scoring_function, greater_is_better=False)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring=custom_scorer, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search
```

---

## Results and Evaluation

### Advanced Text Classification

**Performance Metrics**:

- **Accuracy**: Proportion of correctly classified instances.
- **Precision and Recall**: Evaluated model’s ability to identify relevant instances.
- **F1 Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Visualized performance across classes.

**Comparison of Models**:

- **Logistic Regression**: Achieved 85% accuracy with reasonable precision and recall.
- **SVM**: Achieved 90% accuracy, outperforming Logistic Regression in handling text complexities.

**Visualizations**:

- **Confusion Matrices**: Illustrated types of errors.
- **ROC Curves**: Showed trade-offs between true positive rates and false positive rates.

### Wine Quality Classification

**Model Performance**:

- **Log Loss**: 1.2115421
- **Accuracy**: 96.49%

**Model Comparison**: The tuned SVC model outperformed the default parameter model in both log-loss and accuracy.

```python
svc_tuned = tune_SVC_model(X_train, y_train)
y_pred = svc_tuned.predict(X_test)
print('Log Loss value: ', custom_scoring_function(y_test, y_pred))
print('Accuracy: ', round(accuracy_score(y_test, y_pred), 4))
```

**Performance Metrics**:

- **Log Loss**: 1.2115421
- **Accuracy**: 96.49%

**Visualizations**:

- **Confusion Matrix**: Displayed the types of classification errors.
- **Performance Charts**: Compared model performance before and after tuning.

---

## Business Impact

### Advanced Text Classification

The classification models developed have significant practical implications for automated text analysis, improved decision-making, and cost efficiency by reducing manual text processing efforts.

### Wine Quality Classification

The model’s performance enhances wine quality assessment, streamlining production, and improving product consistency, leading to better customer satisfaction and potential cost savings.

---

## Challenges and Solutions

### Advanced Text Classification

**Imbalanced Data**:

- **Solution**: Implemented SMOTE (Synthetic Minority Over-sampling Technique).

**Text Preprocessing**:

- **Solution**: Developed a robust preprocessing pipeline.

**Model Overfitting**:

- **Solution**: Used cross-validation and regularization techniques.

### Wine Quality Classification

**Data Imbalance**:

- **Solution**: Address

ed through careful preprocessing and appropriate performance metrics.

**Feature Scaling**:

- **Solution**: Standardized features to enhance model performance.

---

## Conclusion and Future Work

### Advanced Text Classification

**Project Summary**: Successfully applied machine learning techniques for high-accuracy text classification. Models like SVM provided excellent performance for complex textual data.

**Future Improvements**:

- **Deep Learning Models**: Explore BERT or LSTM.
- **Real-time Implementation**: Develop a real-time classification system.
- **Additional Features**: Incorporate word embeddings like Word2Vec or GloVe.

### Wine Quality Classification

**Project Summary**: Enhanced wine quality classification through SVC hyperparameter tuning, achieving significant performance improvements.

**Future Improvements**:

- **Additional Features**: Enrich the model’s input with more data.
- **Advanced Models**: Explore ensemble methods or deep learning models.
- **Real-Time Integration**: Implement the model in real-time quality control systems.

---

## Personal Reflection

### Advanced Text Classification

**Skills and Growth**: Gained practical NLP experience, enhanced machine learning understanding, and developed problem-solving skills for data science projects.

### Wine Quality Classification

**Skills and Growth**: Deepened knowledge of hyperparameter tuning and model evaluation, enhancing practical machine learning application skills.

---

## Attachments and References

### Supporting Documents

- **Advanced Text Classification**: Code Repository, Data Files, EDA Reports.
- **Wine Quality Classification**: Project Code and Notebooks.

### References

- **Scikit-Learn Documentation**: [User Guide](https://scikit-learn.org/stable/user_guide.html)
- **Pandas Documentation**: [Pandas](https://pandas.pydata.org/)
- **UCI Machine Learning Repository**: [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **ExploreAI Academy**: [ExploreAI](https://exploreai.com)

This report encapsulates the journey and outcomes of two pivotal projects, underscoring the significant learning and practical implications of advanced text classification and wine quality classification. Each project showcases the intricate process of data analysis, model selection, and performance enhancement, ultimately contributing to a deeper understanding of machine learning applications in diverse fields.
