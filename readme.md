# 🚢 Titanic Survival Prediction

A Machine Learning classification project that predicts whether a passenger survived the Titanic disaster using passenger demographic, travel, and family-related information.

## 📌 Project Overview

The objective of this project is to build a Machine Learning model that predicts Titanic passenger survival and compare multiple classification algorithms to identify the best-performing model.

The project follows an end-to-end Machine Learning workflow:

**Data Understanding → Exploratory Data Analysis → Data Cleaning → Feature Engineering → Preprocessing → Model Training → Hyperparameter Tuning → Model Evaluation → Model Selection**

---

## 🎯 Objective

To predict whether a Titanic passenger survived (`1`) or did not survive (`0`) based on available passenger information.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab
* GitHub

---

## 📊 Dataset

The Titanic dataset contains **891 passenger records**.

Important features include:

* `Pclass` – Passenger class
* `Sex` – Passenger gender
* `Age` – Passenger age
* `SibSp` – Number of siblings/spouses aboard
* `Parch` – Number of parents/children aboard
* `Fare` – Passenger fare
* `Embarked` – Port of embarkation
* `Survived` – Target variable

### Target Variable

```text
0 → Did not survive
1 → Survived
```

---

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify patterns related to survival.

The analysis included:

* Dataset structure and data types
* Missing-value analysis
* Duplicate checking
* Survival distribution
* Survival by gender
* Survival by passenger class
* Survival by embarkation port
* Age distribution
* Fare distribution
* Correlation analysis
* Relationship between age and survival
* Relationship between age and fare

Visualizations were created using **Matplotlib** and **Seaborn**.

---

## 🧹 Data Preprocessing

### Missing Values

Missing values were investigated before model training.

Numerical missing values were handled using **median imputation**, while categorical missing values were handled using **most-frequent imputation**.

The `Cabin` feature was removed because of its high proportion of missing values.

### Feature Scaling

Numerical features were standardized using:

```python
StandardScaler()
```

### Categorical Encoding

Categorical variables were converted into numerical representations using:

```python
OneHotEncoder()
```

A `ColumnTransformer` and `Pipeline` were used to combine preprocessing and model training.

---

## ⚙️ Feature Engineering

Several meaningful features were created to improve model performance.

### 1. Family Size

```python
FamilySize = SibSp + Parch + 1
```

This represents the total number of family members travelling together.

### 2. Is Alone

```python
IsAlone = (FamilySize == 1).astype(int)
```

This identifies whether the passenger was travelling alone.

### 3. Title

The passenger's title was extracted from the `Name` column.

Examples include:

```text
Mr
Mrs
Miss
Master
Rare
```

Rare titles were grouped into a common `Rare` category, while similar titles such as `Mlle`, `Ms`, and `Mme` were standardized.

### 4. Fare By Person

```python
FareByPerson = Fare / FamilySize
```

This represents the approximate fare paid per family member.

### 5. Sex × Passenger Class

A combined categorical feature was created:

```python
Sex_Pclass = Sex + "_" + Pclass
```

Examples:

```text
female_1
female_2
female_3
male_1
male_2
male_3
```

These engineered features were included in the model preprocessing pipeline.

---

## 🤖 Machine Learning Models

Five classification models were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Tuned Decision Tree
4. Random Forest
5. Tuned Random Forest

Hyperparameter tuning was performed using `GridSearchCV` with **5-fold cross-validation** for the Decision Tree and Random Forest models.

---

## 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### Final Model Comparison

| Model                      |   Accuracy |  Precision |     Recall |   F1 Score |
| -------------------------- | ---------: | ---------: | ---------: | ---------: |
| 🥇 **Logistic Regression** | **83.80%** | **84.48%** |     71.01% |     77.17% |
| Decision Tree              |     82.68% |     75.68% | **81.16%** | **78.32%** |
| Tuned Random Forest        |     82.12% |     82.46% |     68.12% |     74.60% |
| Tuned Decision Tree        |     81.56% |     82.14% |     66.67% |     73.60% |
| Random Forest              |     79.33% |     76.67% |     66.67% |     71.32% |

---

## 🏆 Final Model Selection

### Logistic Regression

Logistic Regression was selected as the final model because it achieved the **highest test accuracy and precision** among the evaluated models.

### Final Performance

```text
Accuracy  : 83.80%
Precision : 84.48%
Recall    : 71.01%
F1 Score  : 77.17%
```

The Logistic Regression model achieved the best overall accuracy while maintaining a strong balance between precision, recall, and F1 score.

The Decision Tree achieved a slightly higher recall and F1 score, but Logistic Regression achieved the highest accuracy and precision.

---

## 📊 Confusion Matrix

The final Logistic Regression model produced the following confusion matrix on the test set:

```text
                 Predicted
                 0       1

Actual  0       101      9
        1        20     49
```

This shows that the model correctly classified most passengers in both survival classes.

---

## 🔮 Example Prediction

The trained model was also tested on a new passenger using features such as:

```text
Pclass       : 3
Sex          : female
Age          : 25
SibSp        : 1
Parch        : 0
Fare         : 15.0
Embarked     : S
Title        : Ms
FareByPerson : 7.5
Sex_Pclass   : female_3
```

The model predicted:

```text
[0]
```

which corresponds to:

```text
Did not survive
```

---

## 💡 Key Learnings

Through this project, I learned:

* Exploratory Data Analysis
* Missing-value handling
* Feature engineering
* Numerical feature scaling
* Categorical feature encoding
* Building ML pipelines
* Logistic Regression
* Decision Trees
* Random Forest
* Hyperparameter tuning
* Cross-validation
* Classification metrics
* Confusion matrix analysis
* Model comparison and selection

---

## 📁 Project Structure

```text
titanic-survival-prediction/
│
├── Titanicfinal(2).ipynb
├── README.md
└── dataset/
    └── Titanic-Dataset.csv
```

> If the dataset is not included in the repository, provide the dataset source in the README instead.

---

## 🚀 Future Improvements

Possible future improvements include:

* Further feature engineering
* Feature selection
* ROC-AUC evaluation
* Threshold optimization
* Testing additional classification algorithms
* Model deployment using Streamlit
* Creating an interactive web application for survival prediction

---

## 👩‍💻 Author

**Harini G**

BE Computer Science and Engineering
PSG College of Technology, Coimbatore

---

## ⭐ Conclusion

This project demonstrates a complete Machine Learning workflow for Titanic survival prediction, from data exploration and feature engineering to model training, hyperparameter tuning, and evaluation.

Five classification models were compared, and **Logistic Regression achieved the highest test accuracy of 83.80%**.

The project demonstrates practical understanding of data preprocessing, feature engineering, classification algorithms, model evaluation, and model selection.
