from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import csv, numpy as np, pandas as pd
import os
import matplotlib.pyplot as plt  

# Load the data
data = pd.read_csv(os.path.join("templates", "Training.csv"))
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

svm = SVC(probability=True)
clf_svm = svm.fit(x_train, y_train)

# print("Accuracy: ", clf_svm.score(x_test, y_test))

indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]
dictionary = dict(zip(symptoms, indices))

def dosomething_svm(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1,1)).transpose()
    return svm.predict(user_input_label)

accuracies = []
test_sizes = np.linspace(0.1, 0.9, 9)  

for test_size in test_sizes:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    svm = SVC(probability=True)
    clf_svm = svm.fit(x_train, y_train)
    accuracies.append(clf_svm.score(x_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(test_sizes, accuracies, marker='o', linestyle='-', color='b')
plt.title('SVM Classifier Accuracy vs. Test Size')
plt.xlabel('Test Size')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.show()