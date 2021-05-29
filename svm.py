from sklearn.svm import SVC

import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import pickle

data = pd.read_csv("train_feature.csv")
data = shuffle(data)
print(data.shape)
#print(data.head)

x_train = data.drop('700', axis=1)
y_train = data['700']

svclassifier = SVC(kernel='rbf', C=3, decision_function_shape='ovo')
svclassifier.fit(x_train, y_train)

test_data = pd.read_csv("test_feature.csv")
x_test = test_data.drop('700', axis=1)
y_test = test_data['700']
y_pred = svclassifier.predict(x_test)
print(y_pred)
print(y_test)

print(classification_report(y_test, y_pred))

# Save to file in the current working directory
modle_filename = "svm_classifier.pkl"
with open(modle_filename, 'wb') as file:
    pickle.dump(svclassifier, file)

# Load from file
# with open(modle_filename, 'rb') as file:
#     pickle_model = pickle.load(file)
