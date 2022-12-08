from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import neighbors
from src.model import HyperParamSearch


class Classifiers:
    def SVMClassifier(self, xtrain, ytrain):
        Hyperparam = HyperParamSearch.HyperParamSearch(xtrain, ytrain)
        best_param = Hyperparam.SVMSearch()
        svc = svm.SVC(**best_param).fit(xtrain, ytrain)
        return svc


    def SVMOneAgainstAllClassifier(self, xtrain, ytrain):
        Hyperparam = HyperParamSearch.HyperParamSearch(xtrain, ytrain)
        best_param = Hyperparam.SVMOneAgainstAllSearch()
        svm = LinearSVC(max_iter=100, dual=False, **best_param)
        svm.fit(xtrain, ytrain)
        return svm

    def KNNClassifier(self, xtrain, ytrain):
        Hyperparam = HyperParamSearch.HyperParamSearch(xtrain, ytrain)
        best_param = Hyperparam.KNNSearch()
        knn = neighbors.KNeighborsClassifier(**best_param)
        knn.fit(xtrain, ytrain)
        return knn