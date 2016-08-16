# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 18:04:33 2016

@author: Shubhankar
"""

from sklearn import metrics
from sklearn.grid_search import GridSearchCV

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import xgboost
from xgboost.sklearn import XGBClassifier

models.append(('SVM', SVC))
roc_auc_scorer = metrics.make_scorer(metrics.roc_auc_score, greater_is_better=True,
                             needs_threshold=True) 

# prepare configuration for cross validation test harness
num_folds = 10
num_instances = len(train_[features_1].values)
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression))
models.append(('LDA', LinearDiscriminantAnalysis))
models.append(('KNN', KNeighborsClassifier))
models.append(('CART', DecisionTreeClassifier))
models.append(('NB', GaussianNB))
models.append(('XGB', XGBClassifier))
# evaluate each model in turn


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

model_params = {}
model_params['XGB'] = {'n_estimators': 8}

gs_params = {}
gs_params['XGB'] = [{'max_depth': [4, 6]}
,{'n_estimators': [2,3,5,6]}]

class Algo_Comparison:
    
    def __init__(self, models, model_params=None, model_param_search_grid=None,
                 use_cross_val=True, seed=1, n_jobs=1, scoring = 'accuracy',
                 crossval_n_folds=10, fit_final_models=True):
        self.models = models
        self.model_params = model_params
        self.model_param_search_grid = model_param_search_grid
        self.use_cross_val = use_cross_val
        self.seed = seed
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.crossval_n_folds = crossval_n_folds
        self.fit_final_models = fit_final_models
        
    def _fit_cross_val(self,X, y, model, name):
        from sklearn import cross_validation
        kfold = cross_validation.KFold(n=X.shape[0], n_folds=self.crossval_n_folds, random_state=self.seed)
        cv_results = cross_validation.cross_val_score(model, X, y, cv=kfold, scoring=self.scoring)
        cv_msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        return cv_msg, cv_results
        
    def _fit_grid_search(self, X, y, model, search_param):
        from sklearn.grid_search import (RandomizedSearchCV,GridSearchCV)
        
        mod_gs_cv = GridSearchCV(model, search_param, scoring=self.scoring, n_jobs=self.n_jobs
        , verbose = 0)
        mod_gs_cv_fit = mod_gs_cv.fit(X, y)
        print(mod_gs_cv_fit.best_estimator_)
        return mod_gs_cv_fit.best_estimator_

    def _chk_dict_key(self, dic, key):
        if key not in dic.keys():
            return {}
        else:
            return dic[key]
    
    def fit(self, X, y):
        self.models_ = {}
        self.model_fits_ = {}
        for name, model in self.models:
            init_param = self._chk_dict_key(self.model_params, name)
            model_with_arg = model(**init_param)
            final_model = model_with_arg
            
            if name in self.model_param_search_grid.keys():
                for param_search_grid in self.model_param_search_grid[name]:
                    final_model = self._fit_grid_search(X, y, final_model
                    , param_search_grid)
            self.models_[name] = final_model
            
            if self.fit_final_models == True:
                self.model_fits_[name] = final_model.fit(X,y)
         
        return self

    def score(self, X, y):
        for name, model in self.models:
            if self.use_cross_val == True:            
                print(self._fit_cross_val(X, y, self.models_[name], name)[0])
    
    def score_plot(self, X, y):
        import matplotlib.pyplot as plt
        cv_results = []
        names = []
        for name, model in self.models:
            if self.use_cross_val == True:            
                msg, cv_result = self._fit_cross_val(X, y, self.models_[name], name) 
                print(msg)
                cv_results.append(cv_result)
                names.append(name)
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(cv_results)
        ax.set_xticklabels(names)
        plt.show()                
                

        
        
        
        
        
        
    def _call_with_optional_arguments(func, arg_dict):
        '''
        calls a function with the arguments **kwargs, but only those that the function defines.
        e.g.
    
        def fn(a, b):
            print a, b
    
        call_with_optional_arguments(fn, {'a':1,'b':1,'c':1})  # because fn doesn't accept `c`, it is discarded
        '''
    
        import inspect
        function_arg_names = inspect.getargspec(func).args
    
        for arg in arg_dict.keys():
            if arg not in function_arg_names:
                del arg_dict[arg]
    
        return func(**arg_dict)        
        
        


results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, train_[features_1].values, pd.to_numeric(train_.Business_Sourced).values, cv=kfold, scoring=roc_auc_scorer)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
