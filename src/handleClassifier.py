from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import warnings

# disable sklearn warnings
warnings.filterwarnings("ignore")


class handleClassifier:
    def __init__(self):
        """
        Constructor for the handleClassifier object.
        """
        # get all the available classifiers from sklearn
        self.estimators = all_estimators(type_filter=["classifier", "regressor"])

    def parseClassifiers(self, listClassifiers: list):
        """
        Construct the specified sklearn classifiers object from listClassifiers.

        Args:
            listClassifiers (list): A list of dict containing the name of sklearn classifiers and their
            fitted configuration (see the notebook for an example).

        Returns:
            list: A list of sklearn classifiers object ready to be fitted.
        """
        # the transformed list containing the sklearn classifiers object
        parsedClassifiers = list()
        # list of classifiers name
        classifierNames = [classifierDict["name"] for classifierDict in listClassifiers]

        # loop over all sklearn classifiers
        for estimatorName, estimatorClass in self.estimators:
            searchIdx = -1
            # try to search in classifierNames each imported classifiers
            try:
                searchIdx = classifierNames.index(estimatorName)
            except ValueError:
                pass

            # if one name is found
            if searchIdx != -1:
                # get its dictionary
                classifierDict = listClassifiers[searchIdx]
                config = dict()

                # check if there is a construct config for this classifier
                if "config" in classifierDict:
                    config = classifierDict["config"]
                # try to initialize it
                try:
                    # construct the sklearn object using config
                    estimator = estimatorClass(**config)
                    # get only the rest of the parameters
                    classifierConfig = {
                        k: v
                        for k, v in classifierDict.items()
                        if k != "name" or k != "config"
                    }
                    # append the constructed object and its fitted config
                    parsedClassifiers.append(
                        {"classifier": estimator, "config": classifierConfig}
                    )

                # raise the exception if it is impossible to import it
                except Exception as exception:
                    print("Unable to import:", estimatorName)
                    print(exception)

        return parsedClassifiers

    def fitClassifiers(
        self,
        XTrain: np.array,
        YTrain: np.array,
        listClassifiers: list,
        plotTrainScore: bool = True,
    ):
        """
        Fit all the classifiers in listClassifiers.

        Args:
            XTrain (numpy.array): the training data
            YTrain (numpy.array): the training labels
            listClassifiers (list): A list of dict containing the name of sklearn classifiers and their
            fitted configuration (see the notebook for an example)
            verbose (bool): plot training score

        Raises:
            ValueError: if a preprocess step is not set in listClassifiers for each classifiers
            ValueError: if a feature option is not recognized
            ValueError: if a fitStrategy step is not set in listClassifiers for each classifiers
            ValueError: if a scoring is not set in listClassifiers for each classifiers
            ValueError: if a fitStrategy option is not recognized

        Returns:
            list: Return a list containing fitted pipelines and/or fitted estimators
        """
        # construct the sklearn classifiers object from listClassifiers
        parsedClassifiers = self.parseClassifiers(listClassifiers)
        # name of the classifiers
        nameList = list()
        # training score
        scoreList = list()

        # fit each classifier
        for classifierDict in parsedClassifiers:
            classifier = classifierDict["classifier"]  # get the classifier object
            config = classifierDict["config"]  # get its fitted configuration
            pipelineList = list()  # a list to create the Pipeline

            # get classifier name
            nameList.append(classifier.__class__.__name__)

            # check if the preprocess step is set
            if not "preprocess" in config:
                raise ValueError("A preprocess key need to be set")

            # append into the pipeline list the preprocess step
            if config["preprocess"]:
                pipelineList.append(("preprocess", StandardScaler()))

            # handle the feature step
            if "feature" in config:
                # get its configuration
                feature = config["feature"]
                configFeature = dict()
                if "config" in feature:
                    configFeature = feature["config"]
                # if its a reduction : use a PCA
                if feature["option"] == "reduction":
                    pipelineList.append(("ftr", PCA(**configFeature)))
                # if its a selection : use a SelectFromModel
                elif feature["option"] == "selection":
                    pipelineList.append(("ftr", SelectFromModel(**configFeature)))
                # otherwise raise an error
                else:
                    raise ValueError("A feature step option is not recognized")

            # append at the last the classifier object
            pipelineList.append(("clf", classifier))
            # create the pipeline
            pipe = Pipeline(steps=pipelineList)

            # raise an error if the fitStrategy is not set
            if not "fitStrategy" in config:
                raise ValueError("A fitStrategy key need to be set")

            # get the fitStrategy
            fitStrategy = config["fitStrategy"]

            # raise an error if the scoring is not set
            if not "scoring" in fitStrategy["config"]:
                raise ValueError("A scoring metric key need to be set")

            # get the config of the fitStrategy
            configFitStrategy = fitStrategy["config"]

            # Cross validation
            if fitStrategy["option"] == "CV":
                cv = cross_validate(
                    estimator=pipe, X=XTrain, y=YTrain, **configFitStrategy
                )
                # fit
                classifier.fit(XTrain, YTrain)
                # get training score
                scoreList.append(cv["test_score"].mean())

            # Grid search strat with CV
            elif fitStrategy["option"] == "GridSearch":
                gs = GridSearchCV(estimator=pipe, **configFitStrategy)
                # fit
                gs.fit(XTrain, YTrain)
                # get the pipeline
                classifierDict["classifier"] = gs.best_estimator_
                # get training score
                scoreList.append(gs.best_score_)

            # otherwise raise an error
            else:
                raise ValueError("A fitStrategy option is not recognized")

        # plot
        if plotTrainScore:
            # create a data frame for the plot
            dfTrainingScore = pd.DataFrame(
                {
                    "Training score": scoreList,
                },
                index=nameList,
            )

            # plot
            axes = dfTrainingScore.plot(
                kind="bar",
                rot=0,
                ylabel="Training score",
                figsize=(30, 10),
            )

            # plot bar values
            axes.bar_label(axes.containers[0], label_type="edge")

        # return the fitted classifiers
        return [dictClassifier["classifier"] for dictClassifier in parsedClassifiers]
