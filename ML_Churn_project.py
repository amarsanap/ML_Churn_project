import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix ,classification_report
import pickle



import os
os.listdir(r"C:\Users\Admin\OneDrive\Documents\Pandas")
df= pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.drop(columns=["customerID"], inplace=True)

for col in df.columns:
    print (col,df[col].unique())
    print("-"*50)
df["TotalCharges"]= df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"]= df["TotalCharges"].astype(float)
df.info()
print(df["Churn"].value_counts())
df.head(3)

df["TotalCharges"]= df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"]= df["TotalCharges"].astype(float)
df.info()
#check the class distribution of target column
print(df["Churn"].value_counts())


#understanding the distribution of  numerical feature
def plot_histogram(df,column_name):
    plt.figure(figsize=(5,3))
    sns.histplot(df[column_name],kde=True)
    plt.title(f"distribution of [column_name]")

    col_mean=df[column_name].mean()
    col_median=df[column_name].median()

    plt.axvline(col_mean,color="red",linestyle="--",label="mean")
    plt.axvline(col_median,color="yellow",linestyle="dotted",label="median")

    plt.legend()
    plt.show()
plot_histogram(df,"MonthlyCharges")



# Box plot for numeric feature
def plot_boxplot(df,column_name):
    plt.figure(figsize=(5,3))
    sns.boxplot(y=df[column_name])
    plt.title(f"box plot of [column_name]")
    plt.ylabel(column_name)
    plt.show()
plot_boxplot(df,"MonthlyCharges")

#Heat map - matrix
plt.figure(figsize=(8,6))
sns.heatmap(df[["MonthlyCharges","TotalCharges","tenure"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation of Heatmap")
plt.show()


#understanding the distribution of  categorical feature analysis
#countplot for categorical columns
object_cols=df.select_dtypes(include="object").columns.to_list()
object_cols=["SeniorCitizen"] + object_cols
for col in object_cols:
    plt.figure(figsize=(5,3))
    sns.countplot(x=df[col])
    plt.title(f"count plot of {col}")
    plt.show()


#Label encoding of target column
df["Churn"]=df["Churn"].replace({"Yes":1, "No":0})
# Identifying column with object data type
object_columns =df.select_dtypes(include="object").columns
print(object_columns)
#initilize a dictioinary to save the encoders
encoders={ }
# Apply label encoding and store the encoder ( convert all string values into numeric values)
for column in object_columns:

    label_encoder=LabelEncoder()
    df[column]=label_encoder.fit_transform(df[column])
    encoders[column]=label_encoder
#save the encoder to pickel file
with open("encoders.pkl","wb") as f:
    pickle.dump(encoders,f)
encoders

#Handle the imbalance target clash
#using smote to rectify the imbance.  Before smote we have to  split the data into  trainy data and test data

#splitting the feature and target
X= df.drop(columns=["Churn"])
y=df["Churn"]
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2 , random_state=42)
#Synthetic Minority Oversampling Technique (SMOTE)
smote=SMOTE(random_state=42)
#Oversample it
X_train_smote , y_train_smote = smote.fit_resample(X_train, y_train)
print(y_train_smote.value_counts())
#dictionary of model
models={
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(random_state=42),
    "XGBoost":XGBClassifier(random_state=42)
}
#Dictionary to store cross validation(CV) results
cv_scores= {}
#Perforem 5-fold cross validation for each model
for model_name , model in models.items():
    print("Training {model_name} with default parameters ")
    scores= cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring ="accuracy")
    cv_scores[model_name]=scores
    print(f"[model_name] cross validation accuracy:{np.mean(scores):.2f}")
    print("-"*50)
cv_scores

# train the random forest model(Highest accuracy) with default parameters then find the final data accuracy
rfc=RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote, y_train_smote)

#Evaluate on test data
y_test_pred= rfc.predict(X_test)
print("Accuracy Score:\n", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

#save the trained model as pickle file
with open("Customer_01Churn_model.pkl","wb") as f:
    pickle.dump(rfc,f)
#load the saved model & encoders and feature names
with open("Customer_01Churn_model.pkl","rb") as f:
    model_data= pickle.load(f)
print(model_data)

#Get a dictionary
input_Amar = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 2,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Mailed check',
    'MonthlyCharges': 53.85,
    'TotalCharges': 108.15
}
def predict_churn(input_dict):

    input_df = pd.DataFrame([input_dict])

    with open("encoders.pkl","rb") as f:
        encoders = pickle.load(f)

    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])

    prediction = model_data.predict(input_df)
    probability = model_data.predict_proba(input_df)

    result = "Churn" if prediction[0] == 1 else "No Churn"

    print("Prediction:", result)
    print("Probability:", probability)

predict_churn(input_Amar)

