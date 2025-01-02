# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

# Load the dataset
dataset = pd.read_excel("HousePricePrediction.xlsx")

# Printing first 5 records of the dataset
print(dataset.head(5))

# Data Preprocessing
# Data Shape
print("Dataset Shape:", dataset.shape)

# Identifying categorical, integer, and float columns
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

# Drop Id column as it will not be used for prediction
dataset.drop(['Id'], axis=1, inplace=True)

# Fill missing values in SalePrice column with mean
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())

# Drop rows with any remaining null values
new_dataset = dataset.dropna()

# OneHotEncoder - Convert categorical data to numerical
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()

# Final dataset after encoding categorical columns
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Splitting Dataset into Training and Testing
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Model 1: Support Vector Regression (SVR)
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_pred = model_SVR.predict(X_valid)
svr_mape = mean_absolute_percentage_error(Y_valid, Y_pred)
print("SVM MAPE:", svr_mape)

# Model 2: Random Forest Regressor
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)
rfr_mape = mean_absolute_percentage_error(Y_valid, Y_pred)
print("Random Forest MAPE:", rfr_mape)

# Model 3: Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
lr_mape = mean_absolute_percentage_error(Y_valid, Y_pred)
print("Linear Regression MAPE:", lr_mape)

# Model 4: CatBoost Regressor
cb_model = CatBoostRegressor()
cb_model.fit(X_train, Y_train)
preds = cb_model.predict(X_valid)
cb_r2_score = r2_score(Y_valid, preds)
print("CatBoost R² Score:", cb_r2_score)

# Visualizing Correlation Matrix
# Select only numerical features for correlation analysis
numerical_dataset = dataset.select_dtypes(include=['number'])
plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Visualizing Unique Values in Categorical Features
unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10, 6))
plt.title('No. Unique values of Categorical Features', fontsize=16)
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values, color='skyblue')  # Color changed here
plt.show()

# Distribution of Categorical Features with a different color palette
plt.figure(figsize=(14, 18))  # Adjust the figure size to be more compact
plt.suptitle('Categorical Features: Distribution', fontsize=20, y=1.05)  # Main title with adjusted position
plt.xticks(rotation=90)
index = 1

# Adjust the number of rows and columns in the subplot grid to better fit the categorical features
num_rows = (len(object_cols) // 3) + 1  # Set 3 columns per row for better fit
num_cols = 3  # Fixed columns at 3 to avoid overcrowding

for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(num_rows, num_cols, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y, palette='muted')  # Changed color palette to 'muted'
    plt.title(f'{col} Distribution', fontsize=10, y=0.95)  # Subplot titles with adjusted vertical position
    index += 1

plt.tight_layout()  # Adjusts the layout to prevent overlap
plt.show()
