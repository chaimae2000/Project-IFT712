from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


class handleClassifier:
    def __init__(self, listClassifiers):
        self.fittedClassifiers = list()

        estimators = all_estimators(type_filter=["classifier", "regressor"])
        classifierNames = [classifierDict["name"] for classifierDict in listClassifiers]

        for estimatorName, estimatorClass in estimators:
            searchIdx = -1
            try:
                searchIdx = classifierNames.index(estimatorName)
            except ValueError:
                pass

            if searchIdx != -1:
                classifierDict = listClassifiers[searchIdx]
                config = dict()

                if "config" in classifierDict:
                    config = classifierDict["config"]
                    del classifierDict["config"]
                try:
                    estimator = estimatorClass(**config)
                    del classifierDict["name"]
                    self.fittedClassifiers.append(
                        {"classifier": estimator, "config": classifierDict}
                    )

                except Exception as exception:
                    print("Unable to import:", estimatorName)
                    print(exception)

    def fitClassifiers(self, dfTrain):
        XTrainDef = dfTrain[dfTrain.columns[1:]].to_numpy()
        YTrain = dfTrain[dfTrain.columns[0]].to_numpy()

        for classifierDict in self.fittedClassifiers:
            XTrain = XTrainDef.copy()
            pipelineList = list()
            classifier = classifierDict["classifier"]
            config = classifierDict["config"]

            if "preprocess" in config:
                option = config["preprocess"]
                if option == 1:
                    pipelineList.append(("preprocessor", StandardScaler()))
                elif option == 2:
                    pipelineList.append(("preprocessor", MinMaxScaler()))

            if "featureSelection" in config:
                featureSelection = config["featureSelection"]
                configFeatureSelection = dict()
                if "config" in featureSelection:
                    configFeatureSelection = featureSelection["config"]
                if featureSelection["option"] == 1:
                    pipelineList.append(
                        ("featureSelection", PCA(**configFeatureSelection))
                    )
                elif featureSelection["option"] == 2:
                    pass  # to do

            pipelineList.append(("clf", classifier))
            pipe = Pipeline(steps=pipelineList)

            if not "fitStrategy" in config:
                raise ValueError("A fitStrategy key need to be set")

            fitStrategy = config["fitStrategy"]

            if not "scoring" in fitStrategy["config"]:
                raise ValueError("A scoring metric key need to be set")

            configFitStrategy = fitStrategy["config"]

            if fitStrategy["option"] == 1:
                cv = cross_validate(
                    estimator=pipe, X=XTrain, y=YTrain, **configFitStrategy
                )
                classifier.fit(XTrain, YTrain)
                print(
                    "{}, CV score = {}".format(
                        classifier.__class__.__name__, cv["test_score"].mean()
                    )
                )

            elif fitStrategy["option"] == 2:
                paramGrid = {
                    "clf__" + k: v for k, v in configFitStrategy["param_grid"].items()
                }
                configFitStrategy["param_grid"] = paramGrid
                gs = GridSearchCV(estimator=pipe, **configFitStrategy)
                gs.fit(XTrain, YTrain)
                print(
                    "{}, GridSearchCV best score = {}".format(
                        classifier.__class__.__name__, gs.best_score_
                    )
                )

    def getClassifiers(self):
        return [
            dictClassifier["classifier"] for dictClassifier in self.fittedClassifiers
        ]
