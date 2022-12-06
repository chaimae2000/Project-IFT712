from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class handleData:
    def __init__(self, dataPath):
        self.path = dataPath

    def loadData(self, indexCol):
        return pd.read_csv(self.path, index_col=indexCol)

    def encodeLabels(self, df, labelsCol):
        labels = df[labelsCol].to_numpy()
        le = LabelEncoder().fit(df[labelsCol])
        df[labelsCol] = le.transform(df[labelsCol])
        return labels

    def splitData(self, df, trainSize, stratifyCol):
        dfTrain, dfTest = train_test_split(
            df, train_size=trainSize, stratify=df[stratifyCol], shuffle=True
        )
        return dfTrain, dfTest
