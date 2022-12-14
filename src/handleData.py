from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class handleData:
    def __init__(self, dataPath: str):
        """
        Constructor for the handleData object.

        Args:
            dataPath (str): the path of the CSV file containing the data set
        """
        self.path = dataPath

    def loadData(self, indexCol: str):
        """
        Load the data into a data frame

        Args:
            indexCol (str): the name of the index column

        Returns:
            pd.DataFrame: a data frame
        """
        return pd.read_csv(self.path, index_col=indexCol)

    def encodeLabels(self, df: pd.DataFrame, labelsCol: str):
        """
        Encode the labels as integer

        Args:
            df (pd.DataFrame): the data frame containing the data set
            labelsCol (str): the name of the labels column

        Returns:
            numpy array: the changed labels
        """
        labels = df[labelsCol].to_numpy()
        le = LabelEncoder().fit(df[labelsCol])
        df[labelsCol] = le.transform(df[labelsCol])
        return labels

    def splitData(self, df: pd.DataFrame, trainSize: float, labelsCol: str):
        """
        Split the data into two sets : a training one and a testing one

        Args:
            df (pd.DataFrame): the data frame containing the data set
            trainSize (float): the size of the training set [0.0,1.0]
            labelsCol (str): the name of the column containing the labels

        Returns:
            tuple: a tuple containing the training and testing sets/labels
        """
        # split the data into two sets
        dfTrain, dfTest = train_test_split(
            df, train_size=trainSize, stratify=df[labelsCol], shuffle=True
        )

        # get labels
        YTrain = dfTrain[labelsCol].to_numpy()
        YTest = dfTest[labelsCol].to_numpy()

        # get data
        XTrain = dfTrain.drop([labelsCol], axis=1).to_numpy()
        XTest = dfTest.drop([labelsCol], axis=1).to_numpy()

        return XTrain, YTrain, XTest, YTest
