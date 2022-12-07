from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class Visualize:
    def get_insights(self, model, xtest, ytest):
        ypred_model = model.predict(xtest)
        acc_score = accuracy_score(ytest, ypred_model)
        conf_matrix = confusion_matrix(ytest, ypred_model)
        class_report = classification_report(ytest, ypred_model)
        print('Accuracy:', acc_score)
        print('---------------------------')
        print('Confusion matrix:')
        print(conf_matrix)
        print('---------------------------')
        print('Classification report:')
        print(class_report)
