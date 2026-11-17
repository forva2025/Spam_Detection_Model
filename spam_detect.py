#import relevant packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score   
import seaborn as sns #dynamical visualization
import matplotlib.pyplot as plt #static visualization
import joblib


#load data
data=pd.read_csv("spam_data.csv")

print(data.describe())
#check for null values in the datasets
print(data.isnull().sum())
#print (data)

#specification of features
x = data.drop(columns=['Email No.', 'Prediction']) #exclude the column predictions from the main dataset
y = data['Prediction'] #target

#split the dataset

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#train the logistic model to classify emailsas spam or not spam

model = LogisticRegression()

model.fit(x_train, y_train)


y_pred = model.predict(x_test)

#Evaluate the model accuracy using accurac, confusion matrix, precision, recall, and f1_score

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

print(f"Accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"Recall: {recall}")
print(f"f1: {f1}")


#Visualize the confusion matrix using Seaborn's heatmap

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

#save the model for deployment
joblib.dump(model,"spam_detection_pipeline.joblib")
