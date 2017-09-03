# Classifier benchmark function
import numpy as np
from time import time
from sklearn import metrics


# Benchmarking
def benchmark(clf, X_train, y_train, X_test, y_test, print_scores=True):
    
    t0 = time()
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    train_time = time() - t0
    
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    
    # Training metrics
    train_score = metrics.accuracy_score(y_train, pred_train)
    train_recall = metrics.recall_score(y_train, pred_train)
    train_precision = metrics.precision_score(y_train, pred_train)
    train_f1 = metrics.f1_score(y_train, pred_train)
    
    # Testing metrics
    test_score = metrics.accuracy_score(y_test, pred)
    test_recall = metrics.recall_score(y_test, pred)
    test_precision = metrics.precision_score(y_test, pred)
    test_f1 = metrics.f1_score(y_test, pred)
    auc = metrics.roc_auc_score(y_test, pred)
    conf_matrix = metrics.confusion_matrix(y_test, pred)
    
    # Print results
    if print_scores == True:
        print("_" * 78)
        print("Training: ")
        print(clf)
        print("Training Time: %0.3fs" %train_time)
        print("Classification Report")
        print("Testing Time: %0.3fs" %test_time)
        print("Accuracy: \n\tTrain: %0.3f \n\tTest: %0.3f" 
            %(train_score, test_score))
        print("Recall: \n\tTrain: %0.3f \n\tTest: %0.3f" 
            %(train_recall, test_recall))
        print("Precision: \n\tTrain: %0.3f \n\tTest: %0.3f" 
            %(train_precision, test_precision))
        print("F1 Score: \n\tTrain: %0.3f \n\tTest: %0.3f" 
            %(train_f1, test_f1))
        print("AUC: %0.3f" %auc)
        print("Confusion Matrix: ")
        print(conf_matrix)
        print()
        # Print coefficients, if any
        if hasattr(clf, "coef_"):
            print("Dimensionality: %d" %clf.coef_.shape[1])
    
    clf_descr = str(clf).split('(')[0]
    
    return (clf_descr, auc, test_score, train_score, test_recall, train_recall, 
            test_precision, train_precision, test_f1, train_f1, train_time, test_time)