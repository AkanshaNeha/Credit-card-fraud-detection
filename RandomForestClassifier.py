import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# Function to plot the confusion Matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":

    # load dataset
    data = pd.read_csv("creditcard.csv")

    ''' Verify data import '''
    print(data.head(10))

    ''' describing the data '''
    print(data.shape)
    print(data.describe())

    ''' Understand data imbalance '''
    fraud = data[data['Class'] == 1]
    valid = data[data['Class'] == 0]
    outlierFraction = len(fraud) / float(len(valid))
    print(outlierFraction)
    print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
    print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

    # separating the X and the Y values
    X = data.drop(['Class'], axis=1)
    Y = data["Class"]
    xData = X.values
    yData = Y.values

    # split the data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)

    # building the Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(xTrain, yTrain)
    # predictions
    yPred = rfc.predict(xTest)

    n_outliers = len(fraud)
    n_errors = (yPred != yTest).sum()
    print("The model used is Random Forest classifier")

    cm = confusion_matrix(yTest, yPred)
    print(cm)

    acc = accuracy_score(yTest, yPred)
    print("The accuracy is {:.2f}%".format(acc * 100))

    prec = precision_score(yTest, yPred)
    print("The precision is {:.2f}%".format(prec * 100))

    rec = recall_score(yTest, yPred)
    print("The recall is {:.2f}%".format(rec * 100))

    class_names = np.array(['0', '1'])
    plot_confusion_matrix(cm,class_names)