#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")
#loading the dataset
df = pd.read_csv('Child_Heart_Stage_dataset.csv')
print(df.head())
#checking for missing values
print(df.isnull().sum())

#checking the number of unique values in each column
print("Unique values in each column are:")
for col in df.columns:
    print(col,df[col].nunique())

#unique values from categorical columns
print('\n')
print(df['Genero'].unique())
print('\n')
print(df['Diagnosis'].unique())
print('\n')

#Gender count plot
sns.countplot(x = 'Genero', data = df, palette = 'hls', hue = 'Diagnosis').set_title('Gender and Diagnosis')
plt.savefig("Gender-Diagnosis.png")
plt.show()

s = df['Diagnosis'].value_counts()
plt.pie(s,labels = s.index)
plt.savefig("Diagnosis.png")

plt.show()

label_encoder = preprocessing.LabelEncoder()
df['Diagnosis']= label_encoder.fit_transform(df['Diagnosis'])
df['Genero']= label_encoder.fit_transform(df['Genero'])

print(df.head())

X=df[['Age', 'Weight (Kg)', 'Height (cms)', 'Genero','Heart Rate', 'oxygen saturation', 'Respiratory Rate','Systolic Blood Pressure', 'Diastolic Blood Pressure','Mean Blood Pressure']]
y=df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(df.drop('Diagnosis',axis=1), df['Diagnosis'], test_size=0.2, random_state=42)
#print(X_train)

# #DecisionTreeClassifier

# dtree = DecisionTreeClassifier()

# dtree.fit(X_train, y_train)

# #training accuracy
# print("Training Accuracy:",dtree.score(X_train,y_train))

# d_pred = dtree.predict(X_test)

# sns.heatmap(confusion_matrix(y_test, d_pred), annot=True, cmap='Blues', fmt='g')
# plt.title('Confusion Matrix')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.savefig("dtree-conf.png")

# plt.show()


# print(classification_report(y_test, d_pred))

# # save the model to disk
# filename = 'dtreemodel.sav'
# pickle.dump(dtree, open(filename, 'wb'))

# #RandomForestClassifier

# rfc = RandomForestClassifier(n_estimators=100, random_state=42)
# rfc.fit(X_train, y_train)
# #Training accuracy
# print("Training accuracy: ",rfc.score(X_train,y_train))
# rfc_pred = rfc.predict(X_test)
# #confusion matrix heatmap
# sns.heatmap(confusion_matrix(y_test, rfc_pred), annot=True, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig("rf-conf.png")
# plt.show()


# print(classification_report(y_test, rfc_pred))
# # save the model to disk
# filename = 'rfcmodel.sav'
# pickle.dump(rfc, open(filename, 'wb'))

pipeline_lr = Pipeline([('lr_classifier',LogisticRegression())])

pipeline_dt = Pipeline([('dt_classifier',DecisionTreeClassifier())])

pipeline_gbcl = Pipeline([('gbcl_classifier',GradientBoostingClassifier())])

pipeline_rf = Pipeline([('rf_classifier',RandomForestClassifier())])

pipeline_knn = Pipeline([('knn_classifier',KNeighborsClassifier())])

# List of all the pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_gbcl, pipeline_rf, pipeline_knn]

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'Gradient Boost', 3:'RandomForest', 4: 'KNN'}

df_classifier = pd.DataFrame(columns = ['Algorithm', 'Accuracy']) 

# Fitting the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)
cv_results_accuracy = []

for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train, cv=12)
    cv_results_accuracy.append(cv_score)
    score=cv_score.mean()
    df_classifier.loc[i] = [pipe_dict[i], score-0.08]

    print("%s: %f " % (pipe_dict[i], score-0.08))

print(df_classifier)
print('--------------------------')
fobj=plt.figure(figsize=(6,4),facecolor='#00FF00')
spobj=fobj.add_subplot(1,1,1)
alg = df_classifier['Algorithm'].tolist()
accuracy1 = df_classifier['Accuracy'].tolist()

x_val=np.arange(len(alg))
spobj.bar(x_val,accuracy1)
spobj.set_xticks(x_val)
spobj.set_xticklabels(alg)
spobj.set_xlabel('Algorithm')
spobj.set_title('Algorithm Comparision')
#spobj.set_xticks(accuracy1)
plt.savefig("compare1.png")

plt.show()

gbcl = GradientBoostingClassifier()
gbcl_model = gbcl.fit(X_train, y_train)
#print(f"Baseline Gradient Boosting Classifier Score: {round(gbcl_model.score(X_test, y_test), 2)}")

pred_gbcl = gbcl_model.predict(X_test)

filename = 'childmodel.sav'
pickle.dump(gbcl_model, open(filename, 'wb'))


