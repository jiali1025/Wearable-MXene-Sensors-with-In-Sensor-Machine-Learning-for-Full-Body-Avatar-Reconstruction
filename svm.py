import csv
from sklearn.svm import SVC

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv("train_feature.csv")
data = shuffle(data)
#print(data.shape)
#print(data.head)

x = data.drop('700', axis=1)
y = data['700']
x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size = 0.10)

svclassifier = SVC(kernel='rbf', C=3, decision_function_shape='ovo')
svclassifier.fit(x_train, y_train)

# validation
y_pred = svclassifier.predict(x_validate)
#print(y_pred)
#print(y_validate)
print(classification_report(y_validate, y_pred))

# test prediction
test_data = pd.read_csv("test_feature.csv")
x_test = test_data.drop('700', axis=1)
y_test = test_data['700']
y_pred = svclassifier.predict(x_test)
#print(y_pred)
#print(y_test)

print(classification_report(y_test, y_pred))