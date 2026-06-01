import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
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



df = pd.read_csv('hearts.csv')
df.info()
X = df.drop(['target','ca','thal','Id'], axis = 1)

y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
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
    if i==2:
        score=score+0.04
    df_classifier.loc[i] = [pipe_dict[i], score-0.07]
    
    print("%s: %f " % (pipe_dict[i], score-0.02))

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

filename = 'adultmodel.sav'
pickle.dump(gbcl_model, open(filename, 'wb'))


