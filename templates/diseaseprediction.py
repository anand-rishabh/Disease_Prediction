# Using Decision Tree -----------------------------------

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# import csv, numpy as np, pandas as pd
# import os
# import matplotlib.pyplot as plt  

# data = pd.read_csv(os.path.join("templates", "Training.csv"))
# df = pd.DataFrame(data)
# cols = df.columns
# cols = cols[:-1]
# x = df[cols]
# y = df['prognosis']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# dt = DecisionTreeClassifier()
# clf_dt = dt.fit(x_train, y_train)
# print("Accuracy: ", clf_dt.score(x_test, y_test))
# indices = [i for i in range(132)]
# symptoms = df.columns.values[:-1]
# dictionary = dict(zip(symptoms, indices))

# def dosomething(symptom):
#     user_input_symptoms = symptom
#     user_input_label = [0 for i in range(132)]
#     for i in user_input_symptoms:
#         idx = dictionary[i]
#         user_input_label[idx] = 1

#     user_input_label = np.array(user_input_label)
#     user_input_label = user_input_label.reshape((-1,1)).transpose()
#     return dt.predict(user_input_label)
# # Uncomment below to test the function with example symptoms
# # prediction = dosomething(['headache','muscle_weakness','puffy_face_and_eyes','mild_fever','skin_rash'])
# # print(prediction)
# accuracies = []
# test_sizes = np.linspace(0.1, 0.9, 9)  
# for test_size in test_sizes:
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
#     dt = DecisionTreeClassifier()
#     clf_dt = dt.fit(x_train, y_train)
#     accuracies.append(clf_dt.score(x_test, y_test))
# plt.figure(figsize=(10, 5))
# plt.plot(test_sizes, accuracies, marker='o', linestyle='-', color='b')
# plt.title('Decision Tree Classifier Accuracy vs. Test Size')
# plt.xlabel('Test Size')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.show()  


# Using Logistic Regression --------------------------------------------------------------------------------------------------------------

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import csv, numpy as np, pandas as pd
# import os
# import matplotlib.pyplot as plt  

# data = pd.read_csv(os.path.join("templates", "Training.csv"))
# df = pd.DataFrame(data)
# cols = df.columns
# cols = cols[:-1]
# x = df[cols]
# y = df['prognosis']

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# lr = LogisticRegression(max_iter=1000)
# clf_lr = lr.fit(x_train, y_train)

# print("Accuracy: ", clf_lr.score(x_test, y_test))

# indices = [i for i in range(132)]
# symptoms = df.columns.values[:-1]
# dictionary = dict(zip(symptoms, indices))

# def dosomething_logistic(symptom):
#     user_input_symptoms = symptom
#     user_input_label = [0 for i in range(132)]
#     for i in user_input_symptoms:
#         idx = dictionary[i]
#         user_input_label[idx] = 1

#     user_input_label = np.array(user_input_label)
#     user_input_label = user_input_label.reshape((-1,1)).transpose()
#     return lr.predict(user_input_label)

# accuracies = []
# test_sizes = np.linspace(0.1, 0.9, 9)  

# for test_size in test_sizes:
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
#     lr = LogisticRegression(max_iter=1000)
#     clf_lr = lr.fit(x_train, y_train)
#     accuracies.append(clf_lr.score(x_test, y_test))

# plt.figure(figsize=(10, 5))
# plt.plot(test_sizes, accuracies, marker='o', linestyle='-', color='b')
# plt.title('Logistic Regression Classifier Accuracy vs. Test Size')
# plt.xlabel('Test Size')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.show()

# Using Support Vector Machine -----------------------------------------------------------------------------------------------------

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

print("Accuracy: ", clf_svm.score(x_test, y_test))

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
plt.show()

#  Using K- Nearest Neighbours -----------------------------------------------------------------------

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# import csv, numpy as np, pandas as pd
# import os
# import matplotlib.pyplot as plt  

# # Load the data
# data = pd.read_csv(os.path.join("templates", "Training.csv"))
# df = pd.DataFrame(data)
# cols = df.columns
# cols = cols[:-1]
# x = df[cols]
# y = df['prognosis']

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# knn = KNeighborsClassifier(n_neighbors=5)
# clf_knn = knn.fit(x_train, y_train)

# print("Accuracy: ", clf_knn.score(x_test, y_test))

# indices = [i for i in range(132)]
# symptoms = df.columns.values[:-1]
# dictionary = dict(zip(symptoms, indices))

# def dosomething_knn(symptom):
#     user_input_symptoms = symptom
#     user_input_label = [0 for i in range(132)]
#     for i in user_input_symptoms:
#         idx = dictionary[i]
#         user_input_label[idx] = 1

#     user_input_label = np.array(user_input_label)
#     user_input_label = user_input_label.reshape((-1,1)).transpose()
#     return knn.predict(user_input_label)

# accuracies = []
# test_sizes = np.linspace(0.1, 0.9, 9)  

# for test_size in test_sizes:
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
#     knn = KNeighborsClassifier(n_neighbors=5)
#     clf_knn = knn.fit(x_train, y_train)
#     accuracies.append(clf_knn.score(x_test, y_test))

# plt.figure(figsize=(10, 5))
# plt.plot(test_sizes, accuracies, marker='o', linestyle='-', color='b')
# plt.title('KNN Classifier Accuracy vs. Test Size')
# plt.xlabel('Test Size')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.show()

#  Using Naive Bayes -----------------------------------------------------------------------------------------------------

# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# import csv, numpy as np, pandas as pd
# import os
# import matplotlib.pyplot as plt  

# # Load the data
# data = pd.read_csv(os.path.join("templates", "Training.csv"))
# df = pd.DataFrame(data)
# cols = df.columns
# cols = cols[:-1]
# x = df[cols]
# y = df['prognosis']

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# nb = GaussianNB()
# clf_nb = nb.fit(x_train, y_train)

# print("Accuracy: ", clf_nb.score(x_test, y_test))

# indices = [i for i in range(132)]
# symptoms = df.columns.values[:-1]
# dictionary = dict(zip(symptoms, indices))

# def dosomething_nb(symptom):
#     user_input_symptoms = symptom
#     user_input_label = [0 for i in range(132)]
#     for i in user_input_symptoms:
#         idx = dictionary[i]
#         user_input_label[idx] = 1

#     user_input_label = np.array(user_input_label)
#     user_input_label = user_input_label.reshape((-1,1)).transpose()
#     return nb.predict(user_input_label)

# accuracies = []
# test_sizes = np.linspace(0.1, 0.9, 9)  

# for test_size in test_sizes:
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
#     nb = GaussianNB()
#     clf_nb = nb.fit(x_train, y_train)
#     accuracies.append(clf_nb.score(x_test, y_test))

# plt.figure(figsize=(10, 5))
# plt.plot(test_sizes, accuracies, marker='o', linestyle='-', color='b')
# plt.title('Naive Bayes Classifier Accuracy vs. Test Size')
# plt.xlabel('Test Size')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.show()


# -----------------------------------------------------------------------------------------------------------

# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import LabelBinarizer
# import numpy as np, pandas as pd
# import os
# import matplotlib.pyplot as plt  
# import seaborn as sns

# # Load the data
# data = pd.read_csv(os.path.join("templates", "Training.csv"))
# df = pd.DataFrame(data)
# cols = df.columns[:-1]
# x = df[cols]
# y = df['prognosis']

# # Split the data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# # Define models
# models = {
#     "SVM": SVC(probability=True),
#     "Decision Tree": DecisionTreeClassifier(),
#     "KNN": KNeighborsClassifier(),
#     "Logistic Regression": LogisticRegression(max_iter=200),
#     "Naive Bayes": GaussianNB()
# }

# # Label binarizer for multiclass roc_auc_score
# lb = LabelBinarizer()
# lb.fit(y_train)
# y_train_binarized = lb.transform(y_train)
# y_test_binarized = lb.transform(y_test)

# # Evaluate models
# results = {}
# for name, model in models.items():
#     # Cross-validation
#     cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
#     results[name] = {"Cross-Validation Accuracy": cv_scores.mean()}
    
#     # Fit model
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
    
#     if hasattr(model, "predict_proba"):
#         y_prob = model.predict_proba(x_test)
#     else:
#         y_prob = model.decision_function(x_test)
#         if y_prob.ndim == 1:
#             y_prob = np.expand_dims(y_prob, axis=1)

#     # Metrics
#     results[name]["Accuracy"] = accuracy_score(y_test, y_pred)
#     results[name]["Precision"] = precision_score(y_test, y_pred, average='weighted')
#     results[name]["Recall"] = recall_score(y_test, y_pred, average='weighted')
#     results[name]["F1-Score"] = f1_score(y_test, y_pred, average='weighted')

#     if name != "Naive Bayes":
#         roc_auc = roc_auc_score(y_test_binarized, y_prob, average='weighted', multi_class='ovr')
#         results[name]["ROC-AUC"] = roc_auc
#     else:
#         results[name]["ROC-AUC"] = None
    
#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#     plt.title(f"{name} Confusion Matrix")
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()
    
#     # ROC Curve
#     if name != "Naive Bayes":  # Naive Bayes may not support decision_function or predict_proba properly for ROC curve
#         fpr = dict()
#         tpr = dict()
#         for i in range(y_prob.shape[1]):
#             fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
#             plt.plot(fpr[i], tpr[i], label=f'{name} class {i} (AUC = {roc_auc:.2f})')

# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc='best')
# plt.show()

# # Display results
# results_df = pd.DataFrame(results).T
# print(results_df)

# # Plot comparison of metrics
# results_df.drop(columns=["Cross-Validation Accuracy", "ROC-AUC"], inplace=True)
# results_df.plot(kind='bar', figsize=(14, 7))
# plt.title('Comparison of Models')
# plt.xlabel('Metrics')
# plt.ylabel('Score')
# plt.xticks(rotation=45)
# plt.show()

# ---------------------------------------------------------------------------------------------

# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import LabelBinarizer
# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt  
# import seaborn as sns

# # Load the data
# data = pd.read_csv(os.path.join("templates", "Training.csv"))
# df = pd.DataFrame(data)
# cols = df.columns[:-1]
# x = df[cols]
# y = df['prognosis']

# # Split the data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# # Define models
# models = {
#     "SVM": SVC(probability=True),
#     "Decision Tree": DecisionTreeClassifier(),
#     "KNN": KNeighborsClassifier(),
#     "Logistic Regression": LogisticRegression(max_iter=200),
#     "Naive Bayes": GaussianNB()
# }

# # Label binarizer for multiclass roc_auc_score
# lb = LabelBinarizer()
# lb.fit(y_train)
# y_train_binarized = lb.transform(y_train)
# y_test_binarized = lb.transform(y_test)

# # Evaluate models
# results = {}
# fig, axes = plt.subplots(2, 1 + len(models), figsize=(25, 10))

# for idx, (name, model) in enumerate(models.items()):
#     print(f"Evaluating {name}...")
    
#     # Cross-validation
#     cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
#     results[name] = {"Cross-Validation Accuracy": cv_scores.mean()}
    
#     # Fit model
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
    
#     if hasattr(model, "predict_proba"):
#         y_prob = model.predict_proba(x_test)
#     else:
#         y_prob = model.decision_function(x_test)
#         if y_prob.ndim == 1:
#             y_prob = np.expand_dims(y_prob, axis=1)

#     # Metrics
#     results[name]["Accuracy"] = accuracy_score(y_test, y_pred)
#     results[name]["Precision"] = precision_score(y_test, y_pred, average='weighted')
#     results[name]["Recall"] = recall_score(y_test, y_pred, average='weighted')
#     results[name]["F1-Score"] = f1_score(y_test, y_pred, average='weighted')

#     if name != "Naive Bayes":
#         roc_auc = roc_auc_score(y_test_binarized, y_prob, average='weighted', multi_class='ovr')
#         results[name]["ROC-AUC"] = roc_auc
#     else:
#         results[name]["ROC-AUC"] = None
    
#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, idx])
#     axes[0, idx].set_title(f"{name} Confusion Matrix")
#     axes[0, idx].set_xlabel('Predicted')
#     axes[0, idx].set_ylabel('True')
    
#     # ROC Curve
#     if name != "Naive Bayes":  # Naive Bayes may not support decision_function or predict_proba properly for ROC curve
#         fpr = dict()
#         tpr = dict()
#         for i in range(y_prob.shape[1]):
#             fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
#             axes[1, idx].plot(fpr[i], tpr[i], label=f'class {i} (AUC = {roc_auc:.2f})')
#         axes[1, idx].plot([0, 1], [0, 1], 'k--')
#         axes[1, idx].set_xlabel('False Positive Rate')
#         axes[1, idx].set_ylabel('True Positive Rate')
#         axes[1, idx].set_title(f'{name} ROC Curve')
#         axes[1, idx].legend(loc='best')

# # Adjust layout
# plt.tight_layout()
# plt.show()

# # Display results
# results_df = pd.DataFrame(results).T
# print(results_df)

# # Plot comparison of metrics
# fig, ax = plt.subplots(figsize=(14, 7))
# results_df.drop(columns=["Cross-Validation Accuracy", "ROC-AUC"], inplace=True)
# results_df.plot(kind='bar', ax=ax)
# plt.title('Comparison of Models')
# plt.xlabel('Metrics')
# plt.ylabel('Score')
# plt.xticks(rotation=45)
# plt.show()

# --------------------------------------------------------------------------------------  Both supervised and unsupervised learning

# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score, mutual_info_score, silhouette_score
# from sklearn.preprocessing import LabelBinarizer
# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt  
# import seaborn as sns
# from scipy.stats import mode

# # Load the data
# data = pd.read_csv(os.path.join("templates", "Training.csv"))
# df = pd.DataFrame(data)
# cols = df.columns[:-1]
# x = df[cols]
# y = df['prognosis'].astype('category').cat.codes  # Convert labels to integers

# # Split the data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# # Define models
# supervised_models = {
#     "SVM": SVC(probability=True),
#     "Decision Tree": DecisionTreeClassifier(),
#     "KNN": KNeighborsClassifier(),
#     "Logistic Regression": LogisticRegression(max_iter=200),
#     "Naive Bayes": GaussianNB()
# }

# unsupervised_models = {
#     "KMeans": KMeans(n_clusters=len(np.unique(y)), random_state=42),
#     "Agglomerative Clustering": AgglomerativeClustering(n_clusters=len(np.unique(y)))
# }

# # Label binarizer for multiclass roc_auc_score
# lb = LabelBinarizer()
# lb.fit(y_train)
# y_train_binarized = lb.transform(y_train)
# y_test_binarized = lb.transform(y_test)

# # Evaluate supervised models
# results = {}
# fig, axes = plt.subplots(3, len(supervised_models), figsize=(20, 15))

# for idx, (name, model) in enumerate(supervised_models.items()):
#     print(f"Evaluating {name}...")
    
#     # Cross-validation
#     cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
#     results[name] = {"Cross-Validation Accuracy": cv_scores.mean()}
    
#     # Fit model
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
    
#     if hasattr(model, "predict_proba"):
#         y_prob = model.predict_proba(x_test)
#     else:
#         y_prob = model.decision_function(x_test)
#         if y_prob.ndim == 1:
#             y_prob = np.expand_dims(y_prob, axis=1)

#     # Metrics
#     results[name]["Accuracy"] = accuracy_score(y_test, y_pred)
#     results[name]["Precision"] = precision_score(y_test, y_pred, average='weighted')
#     results[name]["Recall"] = recall_score(y_test, y_pred, average='weighted')
#     results[name]["F1-Score"] = f1_score(y_test, y_pred, average='weighted')

#     if name != "Naive Bayes":
#         roc_auc = roc_auc_score(y_test_binarized, y_prob, average='weighted', multi_class='ovr')
#         results[name]["ROC-AUC"] = roc_auc
#     else:
#         results[name]["ROC-AUC"] = None
    
#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, idx])
#     axes[0, idx].set_title(f"{name} Confusion Matrix")
#     axes[0, idx].set_xlabel('Predicted')
#     axes[0, idx].set_ylabel('True')
    
#     # ROC Curve
#     if name != "Naive Bayes":
#         fpr = dict()
#         tpr = dict()
#         for i in range(y_prob.shape[1]):
#             fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
#             axes[1, idx].plot(fpr[i], tpr[i], label=f'class {i} (AUC = {roc_auc:.2f})')
#         axes[1, idx].plot([0, 1], [0, 1], 'k--')
#         axes[1, idx].set_xlabel('False Positive Rate')
#         axes[1, idx].set_ylabel('True Positive Rate')
#         axes[1, idx].set_title(f'{name} ROC Curve')
#         axes[1, idx].legend(loc='best')

# # Evaluate unsupervised models
# unsupervised_results = {}
# for idx, (name, model) in enumerate(unsupervised_models.items()):
#     print(f"Evaluating {name}...")
    
#     # Fit model
#     model.fit(x)
#     y_pred = model.labels_ if hasattr(model, 'labels_') else model.predict(x)
    
#     # Map cluster labels to true labels
#     labels = np.zeros_like(y)
#     for i in range(len(np.unique(y))):
#         mask = (y_pred == i)
#         labels[mask] = mode(y[mask])[0]
    
#     # Metrics
#     unsupervised_results[name] = {
#         "Adjusted Rand Index": adjusted_rand_score(y, labels),
#         "Mutual Information": mutual_info_score(y, labels),
#         "Silhouette Score": silhouette_score(x, y_pred)
#     }
    
#     # Confusion matrix
#     cm = confusion_matrix(y, labels)
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2, idx])
#     axes[2, idx].set_title(f"{name} Confusion Matrix")
#     axes[2, idx].set_xlabel('Predicted')
#     axes[2, idx].set_ylabel('True')

# # Adjust layout
# plt.tight_layout()
# plt.show()

# # Display results
# results_df = pd.DataFrame(results).T
# unsupervised_results_df = pd.DataFrame(unsupervised_results).T
# print("Supervised Learning Results:")
# print(results_df)
# print("\nUnsupervised Learning Results:")
# print(unsupervised_results_df)

# # Plot comparison of metrics
# fig, ax = plt.subplots(figsize=(14, 7))
# results_df.drop(columns=["Cross-Validation Accuracy", "ROC-AUC"], inplace=True)
# results_df.plot(kind='bar', ax=ax)
# plt.title('Comparison of Supervised Models')
# plt.xlabel('Metrics')
# plt.ylabel('Score')
# plt.xticks(rotation=45)
# plt.show()

# # Plot comparison of unsupervised metrics
# fig, ax = plt.subplots(figsize=(14, 7))
# unsupervised_results_df.plot(kind='bar', ax=ax)
# plt.title('Comparison of Unsupervised Models')
# plt.xlabel('Metrics')
# plt.ylabel('Score')
# plt.xticks(rotation=45)
# plt.show()
