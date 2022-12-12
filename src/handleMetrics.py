from sklearn.model_selection import LearningCurveDisplay
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class handleMetrics:
    def __init__(self):
        """
        Constructor for the handleMetrics object.
        """
        pass
       
    def plotBarChart(self, XTest: np.array, YTest: np.array, classifierList: list):
        """
        Plot the performance metrics as a bar plot

        Args:
            XTest (numpy.array): the testing data
            YTest (numpy.array): the testing labels
            classifierList (list): a list containing fitted sklearn classifier objects
        """
        # lists
        nameList, accuracyList, precisionList, recallList, f1ScoreList, AUCList = (
            [] for i in range(6)
        )

        # loop over each classifier
        for clf in classifierList:
            # predict
            YPred = clf.predict(XTest)
            # get prediction prob if available
            probPred = (
                clf.predict_proba(XTest).tolist()
                if hasattr(clf, "predict_proba")
                else None
            )

            # get name of the classifier
            if clf.__class__.__name__ == "Pipeline":
                nameList.append(clf[-1].__class__.__name__)
            else:
                nameList.append(clf.__class__.__name__)

            # compute performance metrics
            accuracyList.append(accuracy_score(YTest, YPred))
            precisionList.append(precision_score(YTest, YPred, average="macro"))
            recallList.append(recall_score(YTest, YPred, average="macro"))
            f1ScoreList.append(f1_score(YTest, YPred, average="macro"))
            AUCList.append(
                roc_auc_score(YTest, probPred, multi_class="ovr")
                if isinstance(probPred, list)
                else None
            )

        # create a data frame for the plot
        dfClassifierMetric = pd.DataFrame(
            {
                "Accuracy": accuracyList,
                "Precision": precisionList,
                "Recall": recallList,
                "F1-score": f1ScoreList,
                "AUC": AUCList,
            },
            index=nameList,
        )

        # plot
        axes = dfClassifierMetric.plot(
            kind="bar",
            rot=0,
            subplots=True,
            sharex=False,
            ylabel="Score",
            figsize=(30, 30),
        )
        
        # plot bar values
        for ax in axes:
            ax.bar_label(ax.containers[0], label_type='edge')

    def plotLearningCurve(self, df : pd.DataFrame, labelsCol: str, classifierList : list):
        """
        Plot learning curves

        Args:
            df (pd.DataFrame): data set
            labelsCol (str): the name of the column containing the labels
            classifierList (list): a list containing fitted sklearn classifier objects
        """
        # split the data and the labels
        Y = df[labelsCol].to_numpy()
        X = df.drop([labelsCol], axis=1).to_numpy()
        
        numberClf = len(classifierList) # number of classifier
        nRows = int(np.ceil(numberClf / 4)) # number of rows for the subplots

        # create subplots
        fig, ax = plt.subplots(nrows=nRows, ncols=4, figsize=(14, 10), sharey=True)
        fig.tight_layout(pad=3.5)
        ax = ax.flatten()

        # LearningCurve parameters
        param = {
            "X": X,
            "y": Y,
            "train_sizes": np.linspace(0.1, 1.0, 10),
            "score_type": "both",
            "n_jobs": -1,
            "line_kw": {"marker": "o"},
            "std_display_style": "fill_between",
            "score_name": "Accuracy",
        }

        # compute the learning curve for each classifier
        for axIdx, estimator in enumerate(classifierList):
            LearningCurveDisplay.from_estimator(estimator, **param, ax=ax[axIdx])
            handles, _ = ax[axIdx].get_legend_handles_labels()
            ax[axIdx].legend(handles[:2], ["Training Score", "Test Score"])
            estimatorName = estimator.__class__.__name__

            if estimatorName == "Pipeline":
                estimatorName = estimator[-1].__class__.__name__
                
            ax[axIdx].set_title(estimatorName)

        # remove unnecessary plot
        toRemove = (nRows * 4) - numberClf

        for i in range(toRemove):
            ax.flat[(nRows * 4) - 1 - i].set_visible(False)



