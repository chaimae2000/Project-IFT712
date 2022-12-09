import sys
from src.data.load_data import DataLoader
from src.model.classifiers import Classifiers
from src.visualization.visualize import Visualize
sys.path.insert(0, 'src')
def main():
    if len(sys.argv) < 2:
        usage = "\n Usage: python main.py nom_noyau insights\
            \n\n\t nom_noyau: 1 SVMLinear, 2 SVM One-Against-all, 3 KNN\
            \n\t insights, with or without metrics visualization\\n"
        print(usage)
        return

    nom_noyau = int(sys.argv[1])
    insights = int(sys.argv[2])

    dataloader = DataLoader()
    xtrain, xtest, ytrain, ytest = dataloader.load_data_splitted()
    classifiers = Classifiers()
    if nom_noyau == 1:
        clf = classifiers.SVMClassifier(xtrain, ytrain)
    elif nom_noyau == 2:
        clf = classifiers.SVMOneAgainstAllClassifier(xtrain, ytrain)
    elif nom_noyau == 3:
        clf = classifiers.KNNClassifier(xtrain, ytrain)
    print(clf)
    if insights:
        Visualize().get_insights(clf, xtest, ytest)

if __name__ == "__main__":
    main()
