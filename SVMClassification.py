import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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
    ''' Import the data set'''
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

    ''' We need to scale the data'''
    sc = StandardScaler()

    '''We will make use of Support vector machine classifier'''
    classifier = SVC(C=10, kernel='rbf', random_state=0)

    '''We will try the model on entire data set'''
    data["scaled_Amount"] = sc.fit_transform(data.iloc[:, 29].values.reshape(-1, 1))
    data = data.drop(["Time", "Amount"], axis=1)

    label = data.iloc[:, 28]
    data = data.drop(["Class"], axis=1)

    '''We will split the data into training and test set keeping the test set size at 25% of the whole set'''
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=0)

    classifier.fit(X_train, np.array(y_train).ravel())
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("The accuracy is " + str((cm[1, 1] + cm[0, 0]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]) * 100) + " %")
    print("The recall is " + str(cm[1, 1] / (cm[1, 0] + cm[1, 1]) * 100) + " %")
    print("The precision is " + str(cm[1, 1] / (cm[0, 1] + cm[1, 1]) * 100) + " %")

    class_names = np.array(['0', '1'])
    plot_confusion_matrix(cm, class_names)
