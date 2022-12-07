import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self):
        self.xtrain = pd.read_csv("data/raw/train.csv", index_col='id')
        self.ytrain = pd.read_csv("data/raw/test.csv", index_col='id')
    def load_data(self):
        return self.xtrain, self.ytrain

    def load_data_splitted(self):
        labelencoder = LabelEncoder()
        self.xtrain['species'] = labelencoder.fit_transform(self.xtrain['species'])
        xtrain, xtest, ytrain, ytest = train_test_split(self.xtrain.iloc[:, 2:195], self.xtrain['species'],
                                                        train_size=0.5, stratify=self.xtrain['species'])
        return xtrain, xtest, ytrain, ytest
