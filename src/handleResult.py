from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from IPython.display import display
import pandas as pd

class handleResult:
    def __init__(self):
        self.dfClassifierMetric = pd.DataFrame()
        
    def predictionResult(self, dfTest, classifierList):
        """_summary_

        Args:
            dfTest (_type_): _description_
            classifierList (_type_): _description_
        """
        nameList, accuracyList, precisionList, recallList, f1ScoreList, AUCList = ([] for i in range(6))
        for clf in classifierList:
            XTest = dfTest[dfTest.columns[1:]].to_numpy()
            YTest = dfTest[dfTest.columns[0]].to_numpy()
            YPred = clf.predict(XTest)
            probPred = clf.predict_proba(XTest)
            if clf.__class__.__name__ == "Pipeline":
                nameList.append(clf[-1].__class__.__name__)
            else:
                nameList.append(clf.__class__.__name__)
            accuracyList.append(accuracy_score(YTest, YPred))
            precisionList.append(precision_score(YTest, YPred, average='macro')  )          
            recallList.append(recall_score(YTest, YPred, average='macro'))
            f1ScoreList.append(f1_score(YTest, YPred, average='macro'))
            AUCList.append(roc_auc_score(YTest, probPred, multi_class='ovr'))
        self.dfClassifierMetric = pd.DataFrame({"Accuracy":accuracyList, "Precision":precisionList, 
                                           "Recall":recallList, "F1-score":f1ScoreList, "AUC":AUCList}, index=nameList)
        display(self.dfClassifierMetric)

    def plotBarChart(self):
        self.dfClassifierMetric.plot(kind='bar', rot=0, subplots=True, sharex=False, ylabel="Score", figsize=(20,20))
