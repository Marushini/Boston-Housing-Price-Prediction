House Price Prediction - Data Analysis and Machine Learning

This project is designed to predict house prices based on various features in the dataset. It includes exploratory data analysis (EDA), data preprocessing, and model training using different machine learning algorithms, such as Support Vector Regression (SVR), Random Forest Regressor, Linear Regression, and CatBoost Regressor.

Table of Contents

1. [Project Overview](project-overview)
2. [Technologies Used](technologies-used)
3. [Dataset](dataset)
4. [Data Preprocessing](data-preprocessing)
5. [Machine Learning Models](machine-learning-models)
6. [Visualization](visualization)
7. [Results](results)
8. [Installation](installation)

Project Overview

The goal of this project is to predict house prices using machine learning models based on a dataset containing various features of houses. We first perform exploratory data analysis (EDA) to understand the data, followed by preprocessing and feature engineering. Then, we train multiple regression models and evaluate their performance.

Technologies Used

- Python (v3.11)
- Pandas: Data manipulation and analysis
- Matplotlib: Visualization
- Seaborn: Data visualization
- Scikit-learn: Machine learning models and preprocessing
- CatBoost: Gradient boosting machine learning model
- Jupyter Notebooks: For running and testing the code (if applicable)

 Dataset

The dataset used for this project is the "HousePricePrediction.xlsx" file, which contains information about house sales, including columns such as `MSSubClass`, `LotArea`, `OverallQual`, and `SalePrice`.

Data Preprocessing

- Handling Missing Values: Missing values in the `SalePrice` column are filled with the mean value of the column. Rows with any other missing values are dropped.
- One-Hot Encoding: Categorical variables are converted into numerical data using OneHotEncoder.
- Feature Selection: Irrelevant features like the `Id` column are dropped from the dataset.
- Splitting Data: The dataset is split into training and validation sets (80% training, 20% validation).

 Machine Learning Models

We use four regression models to predict house prices:

1. Support Vector Regression (SVR): A machine learning algorithm that fits the model based on the data, minimizing the error margin.
2. Random Forest Regressor: An ensemble method that builds multiple decision trees and combines their results.
3. Linear Regression: A simpler approach that assumes a linear relationship between the features and the target variable.
4. CatBoost Regressor: A powerful gradient boosting method designed to handle categorical data efficiently.

Each model is trained on the training set and evaluated using the Mean Absolute Percentage Error (MAPE) for the first three models and R² score for the CatBoost model.

Visualization

The following visualizations are included in the analysis:

1. Correlation Heatmap: A heatmap showing the correlation between different numerical features in the dataset.
2. Unique Values in Categorical Features: A bar chart displaying the number of unique values in each categorical feature.
3. Distribution of Categorical Features: A set of bar plots that shows the distribution of categorical variables in the dataset.

 Results

The performance of each model is evaluated based on MAPE and R² scores. Here's a brief summary of the results:

- SVR**: Evaluated using Mean Absolute Percentage Error (MAPE).
- Random Forest Regressor: Evaluated using MAPE.
- Linear Regression: Evaluated using MAPE.
- CatBoost Regressor: Evaluated using R² score.

The best model based on R² score or MAPE can be selected depending on the project's requirements.

 Installation

To get started with this project, you need to install the required libraries. You can do so by running the following commands:

```bash
pip install pandas matplotlib seaborn scikit-learn catboost
```

After installing the required libraries, you can download the `HousePricePrediction.xlsx` file and run the script.

---

 How to Run

1. Download the `HousePricePrediction.xlsx` dataset and place it in the same directory as the script.
2. Run the script by executing:

```bash
python project.py
```

This will output the results, including model performance metrics and visualizations.

---

Contributing

Feel free to fork this repository and submit a pull request if you have any suggestions or improvements. If you find a bug or have any questions, please open an issue.

