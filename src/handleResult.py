from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from IPython.display import display
import pandas as pd


class handleResult:
    def __init__(self):
        """
        Constructor for the handleResult object.
        """
        # a data frame containing the prediction performance
        self.dfClassifierMetric = pd.DataFrame()

    def predictionResult(self, dfTest: pd.DataFrame, classifierList: list):
        """
        Plot prediction performance using a data frame

        Args:
            dfTest (DataFrame): the testing set (containing also the labels)
            classifierList (list): a list containing fitted sklearn classifier objects
        """
        # lists
        nameList, accuracyList, precisionList, recallList, f1ScoreList, AUCList = (
            [] for i in range(6)
        )

        # get labels (first column)
        XTest = dfTest[dfTest.columns[1:]].to_numpy()
        YTest = dfTest[dfTest.columns[0]].to_numpy()

        # loop over each classifier
        for clf in classifierList:
            # predict
            YPred = clf.predict(XTest)
            # get prediction prob
            probPred = clf.predict_proba(XTest)

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
            AUCList.append(roc_auc_score(YTest, probPred, multi_class="ovr"))

        # create the data frame
        self.dfClassifierMetric = pd.DataFrame(
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
        display(self.dfClassifierMetric)

    def plotBarChart(self):
        """
        Plot the performance metrics as a bar plot
        """
        self.dfClassifierMetric.plot(
            kind="bar",
            rot=0,
            subplots=True,
            sharex=False,
            ylabel="Score",
            figsize=(20, 20),
        )
