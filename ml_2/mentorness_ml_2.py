#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Load the dataset
df = pd.read_csv('/home/hgidea/Desktop/Coding/Python/internship/mentorness/ml_2/FastagFraudDetection.csv')
#%% md
# # 1. Data Exploration
#%% md
# **Basic Data Inspection**
#%%
df.nunique()
#%%
df.head()
#%%
df.info()
#%%
df.describe()
#%%
df.dtypes
#%%
# Check for missing values
print(df.isnull().sum())
#%%
df_dropna = df.dropna(subset=['FastagID'])  # Drop rows with missing values in 'FastagID'

#%%

print(df_dropna.shape)  # Check the new DataFrame shape to see how many rows were dropped
#%%
# Fraud Prevalence
# Convert 'Fraud_indicator' to numeric (assuming 'Fraud' is 1 and 'Not Fraud' is 0)
df['Fraud_indicator'] = df['Fraud_indicator'].replace({'Fraud': 1, 'Not Fraud': 0})

fraud_percentage = df['Fraud_indicator'].mean() * 100
print(f"Fraudulent transactions: {fraud_percentage:.2f}%")
#%%
df.columns
#%% md
# **Data Visualization**
#%%
# Analyze feature distributions (histograms)
df.hist(figsize=(12, 8))
plt.show()
#%%
# Investigate correlations with the target variable (fraud indicator)
# Convert date/time columns to numeric representation before calculating correlations
df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Replace 'Date_Column' with the actual column name
df['Date_Column_Numeric'] = df['Timestamp'].astype(int)  # Convert to numeric timestamp

# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr(method="spearman")  # Now calculate correlations

print(correlation_matrix)
#%%
# Visualize categorical feature relationships (boxplots)
df.boxplot(by="Vehicle_Type", column="Transaction_Amount")
plt.show()
#%%
numerical_features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed']
df[numerical_features].hist(bins=30, figsize=(10, 7))
plt.show()
#%%
# Box plots for numerical features
plt.figure(figsize=(10, 7))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=feature, data=df)
plt.show()
#%%
# Bar plots for categorical features
categorical_features = ['Vehicle_Type', 'Lane_Type', 'TollBoothID']
plt.figure(figsize=(15, 7))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(1, 3, i)
    df[feature].value_counts().plot(kind='bar')
    plt.title(feature)
plt.show()
#%%
#Count of Fraud and Non_Fraud Indicators bold text
sns.countplot(x='Fraud_indicator', data=df, palette=['red', 'green'])
plt.xlabel('Fraud Indicator')
plt.ylabel('Count')
plt.title('Count of Fraud and Non-Fraud Indicators')
plt.show()
#%%
# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
#%%
# Distribution of Transaction Amount by Fraud Indicator
sns.boxplot(
    x = "Fraud_indicator",
    y = "Transaction_Amount",
    showmeans=True,
    data=df,
    palette=["red", "green"]
)

plt.xlabel("Fraud Indicator")
plt.ylabel("Transaction Amount")
plt.title("Distribution of Transaction Amount by Fraud Indicator")
plt.xticks(rotation=45)
plt.show()
#%%
print("\nUnique values in categorical columns:")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].unique()}")
sns.pairplot(df)
plt.show()
#%% md
# # 2.FEATURE ENGINEERING
#%%
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#%%
# Extract additional features from Timestamp
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month
#%%
# Encode categorical features
# Use separate encoder instances for each feature
encoder_vehicle = OneHotEncoder(sparse=False)
encoded_vehicle_type = encoder_vehicle.fit_transform(df[['Vehicle_Type']])

encoder_lane = OneHotEncoder(sparse=False)
encoded_lane_type = encoder_lane.fit_transform(df[['Lane_Type']])

encoder_tollbooth = OneHotEncoder(sparse=False)
encoded_tollbooth_id = encoder_tollbooth.fit_transform(df[['TollBoothID']])

#%%
# Convert to DataFrame and concatenate with the main dataframe
encoded_vehicle_type_df = pd.DataFrame(encoded_vehicle_type, columns=encoder_vehicle.get_feature_names_out(['Vehicle_Type']))
encoded_lane_type_df = pd.DataFrame(encoded_lane_type, columns=encoder_lane.get_feature_names_out(['Lane_Type']))
encoded_tollbooth_id_df = pd.DataFrame(encoded_tollbooth_id, columns=encoder_tollbooth.get_feature_names_out(['TollBoothID']))

#%%
df = pd.concat([df, encoded_vehicle_type_df, encoded_lane_type_df, encoded_tollbooth_id_df], axis=1)

#%%
# Drop original categorical columns
df.drop(['Vehicle_Type', 'Lane_Type', 'TollBoothID', 'Timestamp'], axis=1, inplace=True)

#%%
# Additional feature engineering
df['Transaction_Discrepancy'] = df['Transaction_Amount'] - df['Amount_paid']

#%%
# Drop Vehicle_Dimensions column
df.drop('Vehicle_Dimensions', axis=1, inplace=True)

#%%
# Drop irrelevant columns
df.drop(['Transaction_ID', 'FastagID', 'Vehicle_Plate_Number', 'Geographical_Location'], axis=1, inplace=True)

#%%
# Check final dataset columns
print(df.columns)
#%%
df.head()
#%%
df.shape
#%%
num_col = df.select_dtypes(include = ['int','float','uint']).columns
num_col

#%%
cat_col = df.select_dtypes(include = ['object']).columns
cat_col
#%%
df.nunique()
#%%
df[num_col].nunique()
#%% md
# # 3. Model Development
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#%%
# Split the dataset into training and testing sets
X = df.drop('Fraud_indicator', axis=1)
y = df['Fraud_indicator']
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#%%
# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(class_weight='balanced', random_state=42)
}

#%%
# Hyperparameters to tune
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'Support Vector Machine': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}

#%%
# Perform GridSearchCV for each model
best_estimators = {}
for model_name, model in models.items():
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_estimators[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_}")
    print("\n")

#%%

# Evaluate each best model
for model_name, best_model in best_estimators.items():
    print(f"Evaluating {model_name}")
    y_pred = best_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

#%% md
# '''Hyperparameter tuning using GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# '''
# 
#%%
df.isnull().sum()
#%%
import time
model.fit(X_train, y_train)
# Measure prediction latency
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

prediction_time = end_time - start_time
print(f"Prediction time for {len(X_test)} samples: {prediction_time:.2f} seconds")
print(f"Average prediction time per sample: {prediction_time/len(X_test):.6f} seconds")

#%% md
# **Build an ML Pipeline**
#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
#%%
# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Preprocessing step
    ('classifier', GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=100, random_state=42))
])

#%%

# Fit the pipeline
pipeline.fit(X_train, y_train)
#%%
# Evaluate the pipeline
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Predict the test set results
y_pred = pipeline.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
#%%
# Save the pipeline
import joblib
joblib.dump(pipeline, '/home/hgidea/Desktop/Coding/Python/internship/mentorness/ml_2/fastag_fraud_detection_pipeline.pkl')

#%% md
# # 5. Explanatory Analysis
#%%
# Explanatory Analysis with SHAP
import shap

# Create a SHAP explainer
explainer = shap.Explainer(pipeline.named_steps['classifier'], X_train)



#%%
# Calculate SHAP values
shap_values = explainer(X_test)

#%%

# Visualize SHAP values
shap.summary_plot(shap_values, X_test)

#%%

# Detailed explanation for a single prediction
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
#%%
# Assuming 'X_train' is your training data
print(X_train.shape)  # This should output (num_samples, 23)

#%%

#%%
