import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#import missingno as msno
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score

data_df = pd.read_csv("fetal_health.csv")
print(data_df.sample(10))
print(data_df.isna().sum(axis=0)
)

# Evaluating the target column and checking for imbalance of the data, 
colors=["#483D8B","#4682B4", "#87CEFA"]
ax = sns.countplot(data= data_df, x="fetal_health", palette=colors)
ax.bar_label(ax.containers[0])
plt.savefig("target.png")

plt.show()

X=data_df.drop(["fetal_health"],axis=1)
y=data_df["fetal_health"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=25)
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
    df_classifier.loc[i] = [pipe_dict[i], cv_score.mean()]

    print("%s: %f " % (pipe_dict[i], cv_score.mean()))

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
print(f"Baseline Gradient Boosting Classifier Score: {round(gbcl_model.score(X_test, y_test), 2)}")

pred_gbcl = gbcl_model.predict(X_test)

filename = 'fetalmodel.pkl'
pickle.dump(gbcl_model, open(filename, 'wb'))

scores_gbcl = cross_val_score(gbcl, X_train, y_train, cv = 8, n_jobs = 2, scoring = "accuracy")

print(f"CV scores for Gradient Boosting Classifier model:\n{scores_gbcl}")
print(f"CV Mean score: {round(scores_gbcl.mean(), 2)}")
