import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('diabetes.csv')
data.head()

diabetic_patients = data[data.Outcome == 1]
healthy_people = data[data.Outcome == 0]

plt.scatter(healthy_people.Age, healthy_people.Glucose, color = 'green', label='healthy', alpha = 0.4)
plt.scatter(diabetic_patients.Age, diabetic_patients.Glucose, color = 'red', label='diabetic patient', alpha = 0.4)
plt.xlabel('Age')
plt.ylabel('Glucose')
plt.legend()
plt.show()

#Let's determine the x and y axes.
y = data.Outcome.values
x_raw_data = data.drop(["Outcome"], axis = 1)
x = (x_raw_data - np.min(x_raw_data))/(np.max(x_raw_data)-np.min(x_raw_data))

print("Raw data before normalization:\n")
print(x_raw_data.head())

print("\n\n\nThe data we will give to AI for training after normalization:\n")
print(x.head())

#train-test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

#kNN model
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Validation test result of our test data for K=3", knn.score(x_test, y_test))

counter = 1
for k in range(1,18):
    knn_new = KNeighborsClassifier(n_neighbors = k)
    knn_new.fit(x_train, y_train)
    print(counter, " ", "Accuracy rate: % ", knn_new.score(x_test, y_test)*100)
    counter += 1
