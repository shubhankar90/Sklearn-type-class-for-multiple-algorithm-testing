# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:40:05 2016

@author: Shubhankar Mitra
"""

  



import numpy as np
from random import randrange
from sklearn import datasets
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

import xgboost
from xgboost.sklearn import XGBClassifier,XGBRegressor

from algocomparison import AlgoComparison

# prepare models
models = []
models.append(('LR', LogisticRegression))
models.append(('KNN', KNeighborsClassifier))
models.append(('CART', DecisionTreeClassifier))
models.append(('NB', GaussianNB))
models.append(('XGB', (XGBClassifier)))

#Initialize defualt model parameters
model_params = {}
model_params['XGB'] = {'learning_rate':0.1,
 'n_estimators':1000,
 'max_depth':5,
 'min_child_weight':1,
 'gamma':0,
 'subsample':0.8,
 'colsample_bytree':0.8,
 'objective': 'binary:logistic',
 'nthread':4,
 'scale_pos_weight':1,
 'seed':27}

#Initialize model hyper-parameter search space
gs_params = {}
gs_params['XGB'] = [
{
 'max_depth':[i for i in range(3,10,2)],
 'min_child_weight':[i for i in range(1,6,2)]
},
{
 'min_child_weight':[6,8,10,12]
},
{
 'gamma':[i/10.0 for i in range(0,5)]
},
{
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
},
{
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
,
{
'learning_rate': [0.01],
 'n_estimators':[5000]
}
]


#Load Data
iris = datasets.load_iris()
x_train = iris.data[:, :2]  # we only take the first two features. We could
y_train = iris.target


roc_auc_scorer = metrics.make_scorer(metrics.roc_auc_score, greater_is_better=True,
                             needs_threshold=True) 
#Initialize class
multi_algo_comparison = AlgoComparison(models, model_params=model_params, 
                            model_param_search_grid=gs_params
                            , n_jobs = 3, scoring=roc_auc_scorer
                            ,use_cross_val=False)
#Fit models and hyperparameters
multi_algo_comparison_fit = multi_algo_comparison.fit(x_train,y_train)
#Check accuracy and results
multi_algo_comparison_fit.score_plot(x_train,y_train)