import sklearn as sk 
import numpy as np 
import pandas as pd

from sklearn.metrics import accuracy, confusion_matrix

"""
Steps to be followed after data preprocessing:

1. Data splitting into training/validation/holdout paritions 
2. Selection of models to be tested under cross validation 
3. 10-Fold CV on training partition to check for in-sample accuracy 
4. Fit the models on training set and evaluate performance using validation set (still in-sample data)
5. Re-assess models after feature elimination / dimension reduction using 10-Fold CV 
6. Repeat step 4 
7. Optimize the learning models using evolutionary GA search (GridSearch removed due to extremely high complexity time)
8. Re-assess the models after fitting them
9. Generate performance metrics (Confusion matrix, ROC curve, feature importance bar-plot, train-test accuracy graph)
10. Fit these models in an ensemble stack and then assess their combined predicitive metrics
11. Assess the ensemble stack against RNN
"""

class ModelBuilder:
  pass

#Split data into training and holdout set. Further split the training into training and validation sets, X and y
	def __init__(self, X, y):
		self.X = X
		self.y = y 
	
	#Create a training and validation set from the training sample generated before class generation
	def train_test_splitter(self, test_size=0.2):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)
		
	#Cross validation for training set
	def cross_validator(self, X_train, y_train, model, n_folds=10):
		model_cv = cross_val_score(model, X_train, y_train, cv=self.n_folds, scoring='accuracy')
		print('10 fold CV scores: ', model_cv) 
		print('Mean 10-fold CV score: ', np.mean(model_cv))
		
	#Class instance to create a 10-fold CV dataframe of all the classifiers
	def create_cv_dataframe(self, X_train, y_train, X_test, y_test, on_test_data=True, n_folds=10):	
	#Store these values in a dataframe for easy acces to 10-fold CV scores 
		if self.on_test_data == True
			#Empty list to hold CV scores
			xyz = []
			accuracy=[]
			classifiers=['DecisionTree','BaggingClassifier','RandomForests','SVC']
			models = [DecisionTreeClassifier(), BaggingClassifier(), RandomForestClassifier(), SVC()]
			for i in models:
			    model = i
			    cv_result = cross_val_score(model,X_train,y_train, cv = self.n_folds,scoring = "accuracy")
			    cv_result=cv_result
			    xyz.append(cv_result.mean())
			    accuracy.append(cv_result)
			new_models_dataframe2=pd.DataFrame({'CV Mean':xyz},index=classifiers)
		elif self.on_test_data != True:
			#If the data set is chosen as a testing set
			

		
	
	#After cross validation, fit each model with the training dataset and test its metrics using validation set 
	def model_fitter(self, model, X_train, y_train, X_test, y_test, needs_scaling=False):
		#Scaling is required for algorithms which need data to be scaled on a unified interval, notably being SVM and XGBoost
		if self.needs_scaling == False:
			model.fit(X_train, y_train)
			predictions = model.predict(X_test, y_test)
			metric = accuracy(y_test, predictions)
			print('Classification score: ',metric)
			break
		#Import MinMaxScaler module to instantly scale data and fit for models like SVM and XGBoost
		elif self.needs_scaling != False:
			scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
			X_train_scaled = scaler.transform(X_train)
			X_test_scaled = scaler.transform(X_test)
			#Models like SVM or algos which might not be performing well on non-scaled data can be applied here
			model.fit(X_train_scaled, y_train)
			scaled_predictions = model.predict(X_test_scaled, y_test)
			scaled_metric = accuracy(y_test, scaled_predictions)
			print('Classification score on scaled data: ', scaled_metric)
			break
			
		
######################################################################################################################################		
			
"""Stacking module to be made here""" 		
		
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#Final classifier (XGBoost) for the stack 
from xgboost import XGBClassifier

#Metrics 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

#Hyper-parameter tuning
from sklearn.model_selection import GridSearchCV		
		

class EnsembleStack(None):
	pass 

	def initalize_base_classifiers():
		dtree = DecisionTreeClassifier()
		bagger = BaggingClassifier()
		svm = SVC()
		rf = RandomForestClassifier()
		
		#Dictionary to store all models
		models = {
			"dtree": dtree,
			"bagger": bagger,
			"svm": svm,
			"rf": rf}
		return models 
	
	#Takes the model list and fits them using a for-loop iteration 
	def predictor(model_list):
    data = np.zeros((y_test.shape[0], len(model_list)))
    data = pd.DataFrame(data)
    
    print("Fitting the models now...")
    cols = list()
    for i, (name, classifier) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        classifier.fit(X_train, y_train)
        data.iloc[:, i] = classifier.predict_proba(X_test)[:, 1]
        cols.append(name)
        print("Complete")
        
    data.columns = cols
    print("Done. \n")
    return data
	
	#Score each classifier on the basis of AUC score
	def score_models(data, y):
    print("Scoring models")
    for classifier in data.columns:
        score = roc_auc_score(y, data.loc[:, classifier])
        print("%-26s: %.3f" % (classifier, score))
    print("Done.\n")
			
			
			
		

      
