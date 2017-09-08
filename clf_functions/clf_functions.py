# Classifier benchmark function
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from time import time
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
import warnings

def clf_score(clf, x_train, y_train, x_test, y_test):
    ovr_clf = OneVsRestClassifier(clf)
    ovr_clf.fit(x_train, y_train)
    prediction = ovr_clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    
    # There are some categories without predictions 
    # which will raise warnings
    # Suppress those with warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1 = metrics.f1_score(y_test, prediction, 
                              average='weighted')
        precision = metrics.precision_score(y_test, 
                    prediction, average='weighted')
        recall = metrics.recall_score(y_test, 
                      prediction, average='weighted')
    
    return accuracy, f1, precision, recall

def clf_comp(clf_list, X, Y, clf_name=None, n_splits=5, multi_label=False):
    '''
    Classifier Comparison function used to quickly compare classifier
    pipeline performance.
    
    ===========
    Inputs
    clf_list: list of classifier or pipeline objects
    X: training data
    Y: test data
    clf_name: optional list of classifier names. If none, the results
        enumerated
    n_splits: integer number of splits for stratified k-fold cross-validation
    multi_label: if true, the labels are binarized. Default is false.
    
    ===========
    Outputs
    Prints classification metrics to inform model selection:
        accuracy
        f1 score
        precision score
        recall score
        model training and testing time    
    '''
    skf = StratifiedKFold(n_splits=n_splits)
    
    for i, clf in enumerate(clf_list):
        accuracy = []
        f1 = []
        precision = []
        recall = []
        t_0 = time.time()
        for train_index, test_index in skf.split(X, Y):
            # Split test and training data for each fold
            x_train = X[train_index]
            y_train = Y[train_index]
            x_test = X[test_index]
            y_test = Y[test_index]
            
            # Binarize for multi-label data
            if multi_label:
                mlb = MultiLabelBinarizer()
                y_train = mlb.fit_transform(y_train)
                y_test = mlb.fit_transform(y_test)
                
            # Train and score the classifier
            acc_, f1_, prec_, recall_ = clf_score(clf, x_train,
                                                 y_train, x_test, y_test)
            
            accuracy.append(acc_)
            f1.append(f1_)
            precision.append(prec_)
            recall.append(recall_)
        
        t_1 = time.time()
        # Print summary statistics
        acc_mean = np.mean(accuracy)
        f1_mean = np.mean(f1)
        precision_mean = np.mean(precision)
        recall_mean = np.mean(recall)
        
        print("="*20)
        if clf_name:
            print("Classifier: " + clf_name[i])
        else:
            print("Classifier: " + str(i))
        print("Average Metrics")
        print("Accuracy: %.2f" %acc_mean)
        print("F1 Score: %.2f" %f1_mean)
        print("Precision: %.2f" %precision_mean)
        print("Recall: %.2f" %recall_mean)
        print("Total Time %.1f" %(t_1 - t_0))