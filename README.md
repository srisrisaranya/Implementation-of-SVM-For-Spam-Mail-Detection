# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start the Program.

Step 2. Import the necessary packages.

Step 3. Read the given csv file and display the few contents of the data.

Step 4. Assign the features for x and y respectively.

Step 5. Split the x and y sets into train and test sets.

Step 6. Convert the Alphabetical data to numeric using CountVectorizer.

Step 7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

Step 8. Find the accuracy of the model.

Step 9. End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: SARANYA S
RegisterNumber: 212223110044 

```
```
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')
```
![image](https://github.com/user-attachments/assets/92f59edd-c2ba-4061-9dfd-b7cdcabab99b)

```
data.head()
```
![image](https://github.com/user-attachments/assets/cb32783b-da9d-46ce-a97c-1b43ffa7750f)

```
data.tail()
```
![image](https://github.com/user-attachments/assets/46c7ccb8-7c2e-465a-814d-72ea4608237b)

```
data.info()
```
![image](https://github.com/user-attachments/assets/c3be114e-41e8-4401-8095-9eeb8318e3cc)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/313a9093-75c0-4097-aab3-3fcc12d71883)

```
x=data['v2'].values
```
```
y=data['v1'].values
```
```
x.shape
y.shape
```
![image](https://github.com/user-attachments/assets/e128ee56-69c5-4d98-a45f-769938c6ac7d)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
x_train.shape
y_train.shape
```
![image](https://github.com/user-attachments/assets/8d828ef2-e6dd-47d0-939d-4114f582d544)

```
x_test.shape
x_test.shape
```
![image](https://github.com/user-attachments/assets/0ada9295-3aa8-43d4-8e42-ca503f91c097)

```
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
```
```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```
x_train.shape
```
![image](https://github.com/user-attachments/assets/aab4b399-dad2-4dc6-bcc5-ea5f0a11a3c0)

```
x_test.shape
```
![image](https://github.com/user-attachments/assets/21445ff4-b5a3-47a7-b233-abd9ffd19746)

```
type(x_train)
```
![image](https://github.com/user-attachments/assets/17ce8660-c44e-4b53-9e35-a4f6f76f037c)

```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/08336aa5-0e92-404e-bab6-d13f190569e1)

```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![Screenshot 2024-11-06 112911](https://github.com/user-attachments/assets/346ce464-74cc-4d7c-b87b-d8576516ed29)

# Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
