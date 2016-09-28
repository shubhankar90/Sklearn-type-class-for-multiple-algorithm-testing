# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 18:04:33 2016

@author: Shubhankar
"""

from sklearn import metrics
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn import cross_validation
from sklearn.metrics.scorer import check_scoring
import matplotlib.pyplot as plt

class algo_comparison(BaseEstimator):
    
    def __init__(self, models, model_params=None, model_param_search_grid=None,
                 use_cross_val=True, seed=1, n_jobs=1, scoring = None,
                 crossval_n_folds=10, refit=True, para_search_type='GridSearchCV'):
        self.models = models
        self.model_params = model_params
        self.model_param_search_grid = model_param_search_grid
        self.use_cross_val = use_cross_val
        self.seed = seed
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.crossval_n_folds = crossval_n_folds
        self.refit = refit
        self.para_search_type = para_search_type
        
    def _fit_cross_val(self,X, y, model, name):
        kfold = cross_validation.KFold(n=X.shape[0], n_folds=self.crossval_n_folds, random_state=self.seed)
        cv_results = cross_validation.cross_val_score(model, X, y, cv=kfold, scoring=self.scoring)
        return cv_results
        
    def _fit_grid_search(self, X, y, model, search_param):
        if self.para_search_type == 'GridSearchCV':
            mod_gs_cv = GridSearchCV(model, search_param, scoring=self.scoring, n_jobs=self.n_jobs
                                    , verbose = 0)
        else:
            mod_gs_cv = RandomizedSearchCV(model, search_param, scoring=self.scoring, n_jobs=self.n_jobs
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
        if self.model_params is None:
            self.model_params = {}
        if self.model_param_search_grid is None:
            self.model_param_search_grid = {}
        self.models_ = {}
        self.model_fits_ = {}
        for name, model in self.models:
            print('running: ' + name)
            init_param = self._chk_dict_key(self.model_params, name)
            model_with_arg = model(**init_param)
            final_model = model_with_arg
            
            if name in self.model_param_search_grid.keys():
                for param_search_grid in self.model_param_search_grid[name]:
                    final_model = self._fit_grid_search(X, y, final_model
                    , param_search_grid)
            self.models_[name] = final_model
            
            if self.refit == True:
                self.model_fits_[name] = final_model.fit(X,y)
         
        return self

    def predict(self, X):
        result = {}
        for name,model in self.models:
            result[name] = self.model_fits_[name].predict(X)
        return result
    
    def score(self, X, y):
        result = {}
        for name, model in self.models:
            if self.use_cross_val == True:
                result[name] = self._fit_cross_val(X, y, self.models_[name], name)
            else:
                if self.scoring is None:
                    result[name] = self.models_[name].score(X,y)
                else:
                    result[name] = check_scoring(self.models_[name],self.scoring)(self.models_[name],X,y)
        return result
        
    def score_plot(self, X, y):
        cv_results = []
        names = []
        for name, model in self.models:
            cv_result = self._fit_cross_val(X, y, self.models_[name], name) 
            print("%s: %f (%f)" % (name, cv_result.mean(), cv_result.std()))
            cv_results.append(cv_result)
            names.append(name)
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(cv_results)
        ax.set_xticklabels(names)
        plt.show()
        
    def predict_proba(self,X):
        result = {}
        for name, model in self.models:
            try:
                result[name] = self.model_fits_[name].predict_proba(X)
            except:
                pass
        return result
    
