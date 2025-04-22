# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Import pandas
 2. Import Decision tree classifier
 3. Fit the data in the model
 4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: JASWANTH S
RegisterNumber: 212223220037

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("/content/Employee.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())

le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

x = data[["satisfaction_level", "last_evaluation", "number_project", 
          "average_montly_hours", "time_spend_company"]]

y = data["left"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Sample Prediction:", dt.predict([[0.5, 0.8, 9, 260, 6]]))

plt.figure(figsize=(16,10))
plot_tree(dt, feature_names=x.columns, class_names=['Stay', 'Left'], filled=True)
plt.show()
 
*/
```

## Output:

![image](https://github.com/user-attachments/assets/680cd666-849e-4411-929a-7391610cbe21)
![image](https://github.com/user-attachments/assets/596d9019-20f2-448d-91ae-0d517594ddb2)
![image](https://github.com/user-attachments/assets/c5f1ddf5-5be6-4bf7-ae22-a186f81a47bd)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
