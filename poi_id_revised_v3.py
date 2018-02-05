#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
    @Feb 2018
    Udacity project
    #!/usr/bin/python

"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot  as plt 
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from tester import dump_classifier_and_data
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
#algorithm
#DT
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
#from sklearn import svm
#SVC
from sklearn.svm import LinearSVC               
# tuning
#import dataframe as df
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#import dataframe as df
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import NMF

sys.path.append("../tools/")

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_initial = ['poi','shared_receipt_with_poi','salary', 'bonus']
features_list = ['poi',  'shared_receipt_with_poi','salary', 'bonus','fraction_from_poi_email', 'fraction_to_poi_email'] 
feature_all =['poi','bonus','deferral_payments','deferred_income','director_fees','exercised_stock_options','expenses','fraction_from_poi_email','fraction_to_poi_email','from_messages','from_poi_to_this_person','from_this_person_to_poi','loan_advances','long_term_incentive','other','restricted_stock','restricted_stock_deferred','salary','shared_receipt_with_poi','to_messages','total_payments','total_stock_value']
features = ["salary", "bonus"]
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    data = featureFormat(data_dict,features)
    data_initial = featureFormat(data_dict,features_initial)
 #   data_all = featureFormat(data_dict,feature_all)
  #  print ("data is", repr(data))
   
#print (len(data["SKILLING JEFFREY K"].keys()))
# plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary,bonus)
plt.xlabel("Salaries of the executives")
plt.ylabel("Bonuses of the executives")
#plt.show()
# allocation across classes (POI/non-POI)
poi_count = 0
for employee in data_dict:
    if data_dict[employee]['poi'] == True:
        poi_count += 1
print ('the number of person of interest, POI = ', poi_count)
print ('the number of non-POI = ', len(data_dict) - poi_count)

### Task 2: Remove outliers
## Total 
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data_outlier_removed = featureFormat(data_dict, features)
data = data_outlier_removed
"""
list.pop([i])
Remove the item at the given position in the list, and return it. 
If no index is specified, a.pop() removes and returns the last item in the list.
The square brackets around the i in the method signature denote that the parameter is optional,
"""
## NaN
### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['bonus']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
# are they outlier??? 

### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show() 

data_dict_without_new_feature = data_dict
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
### create new features
### new features are: fraction_to_poi_email,fraction_from_poi_email

def dict_to_list(key,normalizer):
    new_list=[]
    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")
# fraction_to_deferred_income_to_total_payment=dict_to_list("deferred_income","total_payments")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]         =fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]           =fraction_to_poi_email[count]
 #   data_dict[i]["fraction_to_deferred_income_to_total_payment"]    =fraction_to_deferred_income_to_total_payment[count]
    count +=1

### store to my_dataset for easy export below
my_dataset = data_dict
data_dict_with_new_feature = data_dict
### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
plt.ylabel("fraction of emails this person sends to poi")
#plt.show()


labels, features = targetFeatureSplit(data)


### Evaluate the performance of a classifier with and without new features
data_with_new_feature = featureFormat(data_dict_with_new_feature,features_list)

labels_initial, features_init = targetFeatureSplit(data_initial)
labels_with_new_feature, features_newf = targetFeatureSplit(data_with_new_feature)
# # 
# data_initial
# data_with_new_feature 
### The performance of the Classifier (PCA) without new feature (initial dataset)
X = features_init
y = labels_initial
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# pca analysis
n_components = 2
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train,y)
X_test_pca = pca.fit_transform(X_test,y)
lr = LogisticRegression()
pipe = Pipeline([('pca', pca), ('logistic', lr)])
pipe.fit(X_train_pca, y_train)
predictions = pipe.predict(X_test_pca)
accuracy_without_new_feature = accuracy_score(y_test, predictions)
precision_without_new_feature = precision_score(y_test, predictions)
recall_without_new_feature = recall_score(y_test, predictions)

### The performance of the Classifier (PCA) with new feature (initial dataset)
X = features_newf
y = labels_with_new_feature
# split into a training and testing set
X_train, X_test, y_train, y_test_nf = train_test_split(X, y, test_size=0.25, random_state=42)
# pca analysis
n_components = 2
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train,y)
X_test_pca = pca.fit_transform(X_test,y)
lr = LogisticRegression()
pipe = Pipeline([('pca', pca), ('logistic', lr)])
pipe.fit(X_train_pca, y_train)
predictions1 = pipe.predict(X_test_pca)
accuracy_with_new_feature = accuracy_score(y_test_nf, predictions1)
precision_with_new_feature = precision_score(y_test_nf, predictions1)
recall_with_new_feature = recall_score(y_test_nf, predictions1)

"""
pred_test = pca.predict(X_test)

print('\nPrediction accuracy for the dataset without new features')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))
pca = PCA(n_components=2)
pca.fit(data)
Data_implementing_PCA = pca.transform(data)   
"""









### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### deploying feature selection
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

## Feature Selection
"""
### use KFold for split and validate algorithm
# GridSearchCV ???
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

# Feature Engineering
X.shape

X_new = SelectKBest(chi2, k=2).fit_transform(X, y)


"""

# PCA
pca = PCA(n_components=2)
pca.fit(data)
Data_implementing_PCA = pca.transform(data)    
"""
clf_LDA = LinearDiscriminantAnalysis(n_components=2)
clf_LDA.fit(features_train, labels_train) #(data = x,y)
Data_implementing_LDA = clf_LDA.transform(features_train)  

# show the features with non null importance, sorted and create features_list of features for the model
features_importance = []
for i in range(len(Data_implementing_LDA.feature_importances_)):
    if Data_implementing_LDA.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], Data_implementing_LDA.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
for f_i in features_importance:
    print (f_i)
features_list = [x[0] for x in features_importance]

importances = clf_LDA.feature_importances_
indices = np.argsort(importances)[::-1]
svm = Pipeline([('scaler',StandardScaler()),("kbest", SelectKBest()),('svm',svm.SVC())])
param_grid = ([{'svm__C': [100],'svm__gamma': [0.1],'svm__degree':[2],'svm__kernel': ['poly'],'kbest__k':['all']}])
clf_ng = GridSearchCV(svm, param_grid, scoring='recall').fit(features, labels).best_estimator_
"""

### Intelligent feature selection
clf_LDA = LinearDiscriminantAnalysis(n_components=2)
clf_LDA.fit(features_train, labels_train) #(data = x,y)
Data_implementing_LDA = clf_LDA.transform(features_train)   

pred_test_LDA = clf_LDA.predict(features_test)
# Show prediction accuracies in intelligent features data.
print('\nPrediction accuracy for the dataset with LDA')
print('{:.2%}\n'.format(metrics.accuracy_score(labels_test, pred_test_LDA)))


### Feature Scaling
RANDOM_STATE = 42
FIG_SIZE = (10, 7) 
# Make a train/test split using 30% test size
# Fit to data and predict using pipelined GNB and PCA.
unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
unscaled_clf.fit(features_train, labels_train)
pred_test = unscaled_clf.predict(features_test)

# Fit to data and predict using pipelined scaling, GNB and PCA.
std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(features_train, labels_train)
pred_test_std = std_clf.predict(features_test)

# Show prediction accuracies in scaled and unscaled data.
print('\nPrediction accuracy for the normal dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(labels_test, pred_test)))

print('\nPrediction accuracy for the standardized dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(labels_test, pred_test_std)))

### Feature importance
#import numpy as np
import matplotlib.pyplot as plt

#from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
X = features_newf
y = labels_with_new_feature
# split into a training and testing set
X_train, X_test, y_train, y_test_nf = train_test_split(X, y, test_size=0.25, random_state=42)

# Build a classification task using 3 informative features
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
"""
# Print the feature ranking
print("-------------Feature ranking:---------------")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

print("-----------End of ranking--------------")
"""

### Step 4: Algorithm
t0 = time()
# Algorithm: Decision tree
clf_DT = DecisionTreeClassifier()
clf_DT.fit(features_train,labels_train)
y_pred_DT = clf_DT.predict(features_test)
precision_DT = precision_score(labels_test, y_pred_DT, average='macro') 
recall_DT = recall_score(labels_test, y_pred_DT, average='macro') 

score_DT = clf_DT.score(features_test,labels_test)
#precision_score(y_true, y_pred, average='macro') 
#precision_DT = precision_score(features_test,labels_test, average='macro')  
#y_score_DT = clf_DT.decision_function(features_test)
# Algorithm: Support Vector Classifier
clf_SVC = LinearSVC()
clf_SVC.fit(features_train,labels_train)
y_pred_SVC = clf_SVC.predict(features_test)
precision_SVC = precision_score(labels_test, y_pred_SVC, average='macro') 
recall_SVC = recall_score(labels_test, y_pred_SVC, average='macro') 

score_SVC_decision_function = clf_SVC.decision_function(features_test)
# Algorithm: KNN
from sklearn import neighbors
clf_KNN = neighbors.KNeighborsClassifier()
clf_KNN.fit(features_train,labels_train)
y_pred_KNN = clf_KNN.predict(features_test)
precision_KNN = precision_score(labels_test, y_pred_KNN, average='macro') 
recall_KNN = recall_score(labels_test, y_pred_KNN, average='macro') 
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
score_RT = clf_KNN.score(features_test,labels_test)


## Import libraries
from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression()
clf_LR.fit(features_train,labels_train)
y_pred_LR = clf_LR.predict(features_test)
precision_LR = precision_score(labels_test, y_pred_LR, average='macro') 
recall_LR = recall_score(labels_test, y_pred_LR, average='macro') 

from sklearn.ensemble import AdaBoostClassifier
clf_AD = AdaBoostClassifier()
clf_AD.fit(features_train,labels_train)
y_pred_AD = clf_AD.predict(features_test)
precision_AD = precision_score(labels_test, y_pred_AD, average='macro') 
recall_AD = recall_score(labels_test, y_pred_AD, average='macro') 


#Compute the average precision score¶
from sklearn.metrics import average_precision_score
average_precision_DT = average_precision_score(labels_test, y_pred_DT)
average_precision_SVC = average_precision_score(labels_test, y_pred_SVC)
average_precision_KNN = average_precision_score(labels_test, y_pred_KNN)
#average_precision_SVC_score = average_precision_score(labels_test, score_SVC)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 

pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', LinearSVC())
])

N_FEATURES_OPTIONS = [1, 3, 5]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]
reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
grid.fit(features_train,labels_train)
y_pred_DT_GridSearchCV = grid.predict(features_test)
mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_gr`id iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
print(classification_report(labels_test, y_pred_DT_GridSearchCV))
   
print("Best parameters set found during tuning:")
print()
print(grid.best_params_)
print("End of tuning")
    
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('POI vs Non-POI classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')


### Validation
    # validation is performed in the TESTER.PY code
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results.
clf = clf_DT
dump_classifier_and_data(clf, my_dataset, features_list)