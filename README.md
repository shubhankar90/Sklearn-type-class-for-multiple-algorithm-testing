# Sklearn-type-class-for-multiple-algorithm-testing
This is an sklearn type class for sequentially testing multiple sklearn interface models on data with hyper-parameter tuning for the individual models. Hyper-parameter tuning is done using grid search. Cross validation is used to choose between models. Hyperparameter search space and sklearn interface compatible models are supplied by the user. Cross validation and grid search settings can be user configurable.

Usage:
1. Supply machine learning models with sklearn compatible interface.
2. Supply hyperparameter search space for the supplied sklearn compatible models.
3. Intantiate class with the model and hyper-parameter list and dictionary data structures.
4. Run fit function of the intantiated class.
5. Run Score and score_plot function to check results for best model and its optimal hyper-parameter combination and other models.

Example Usage Code:

```python

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
multi_algo_comparison = algo_comparison(models, model_params=model_params, 
                            model_param_search_grid=gs_params
                            , n_jobs = 3, scoring=roc_auc_scorer
                            ,use_cross_val=False)
#Fit models and hyperparameters
multi_algo_comparison_fit = multi_algo_comparison.fit(x_train,y_train)
#Check accuracy and results
multi_algo_comparison_fit.score_plot(x_train,y_train)
```

To Do: More usage examples and hosting on Pipi for pip install :)
