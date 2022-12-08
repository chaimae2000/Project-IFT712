from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import neighbors

class HyperParamSearch:
    def __init__(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    def SVMSearch(self):
        parameters_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': list(range(2, 9)),
             'coef0': list(np.linspace(0.000001, 2, num=10))},
            {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': ['scale', 'auto'],
             'gamma': list(np.linspace(0.000001, 2, num=10))},
            {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'coef0': list(np.linspace(0.000001, 2, num=10))}
        ]
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters_grid, cv=5, n_jobs=-1)
        clf.fit(self.xtrain, self.ytrain)
        return clf.best_params_

    
    def SVMOneAgainstAllSearch(self):
        stratifiedKflod = StratifiedKFold(n_splits=5, random_state=11, shuffle=True)
        svm = LinearSVC(max_iter=100, dual=False)
        params = {'C': np.logspace(-3, 3, 7), 'multi_class': ["crammer_singer", "ovr"], 'penalty': ["l1", "l2"]}
        GridSearchCV_svm = GridSearchCV(svm, params, scoring='accuracy', cv=stratifiedKflod)
        GridSearchCV_svm.fit(self.xtrain, self.ytrain)
        return GridSearchCV_svm.best_params_

    def KNNSearch(self):
        skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
        params = {'n_neighbors': np.arange(2, 15)}
        knn = neighbors.KNeighborsClassifier()
        gs_knn = GridSearchCV(knn, params, cv=skf, scoring='accuracy')
        gs_knn.fit(self.xtrain, self.ytrain)
        return gs_knn.best_params_