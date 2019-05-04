import numpy as np
import pandas as pd
import pickle
import argparse
import os

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM

from matplotlib import style
style.use("ggplot")

def train(features = ['0','1','2','3','4','5','6','7','8','9',
									'10','11','12','13','14','15','16','17','18','19',
									'20','21','22','23','24','25','26','27','28','29',
									'30','31','32','33','34','35','36','37','38','39',
									'40','41','42','43','44','45','46','47','48','49',
									'50','51','52','53','54','55','56','57','58','59',
									'60','61','62','63','64','65','66','67','68','69',
									'70','71','72','73','74','75','76','77','78','79',
									'80','81','82','83','84','85','86','87','88','89',
									'90','91','92','93','94','95','96','97','98','99',
									'100','101','102','103','104','105','106','107','108','109',
									'110','111','112','113','114','115','116','117','118','119',
									'120','121','122','123','124','125','126','127']):
    
    data_df = pd.DataFrame.from_csv("encodings.csv")

    data_df = data_df.reindex(np.random.permutation(data_df.index))

    embeddings = np.array(data_df[features].values)
    print(len(embeddings))

    labels = np.array(data_df["Name"].values.tolist())
    # print(labels)
    le = LabelEncoder().fit(labels)
    # print(le)
    labelsNum = le.transform(labels)
    # print(labelsNum)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    if args.classifier == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)
    elif args.classifier == 'GridSearchSvm':
        print("""
        Warning: In our experiences, using a grid search over SVM hyper-parameters only
        gives marginally better performance than a linear SVM with C=1 and
        is not worth the extra computations of performing a grid search.
        """)
        param_grid = [
            {'C': [1, 10, 100, 1000],
             'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
             'gamma': [0.001, 0.0001],
             'kernel': ['rbf']}
        ]
        clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
    elif args.classifier == 'GMM':  # Doesn't work best
        clf = GMM(n_components=nClasses)

    # ref:
    # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
    elif args.classifier == 'RadialSvm':  # Radial Basis Function kernel
        # works better with C = 1 and gamma = 2
        clf = SVC(C=1, kernel='rbf', probability=True, gamma=2, )
    elif args.classifier == 'DecisionTree':  # Doesn't work best
        clf = DecisionTreeClassifier(max_depth=20)
    elif args.classifier == 'GaussianNB':
        clf = GaussianNB()

    # ref: https://jessesw.com/Deep-Learning/
    elif args.classifier == 'DBN':
        from nolearn.dbn import DBN
        clf = DBN([embeddings.shape[1], 100, 5],  # i/p nodes, hidden nodes, o/p nodes
                  learn_rates=0.1,
                  # Smaller steps mean a possibly more accurate result, but the
                  # training will take longer
                  learn_rate_decays=0.99,
                  # a factor the initial learning rate will be multiplied by
                  # after each iteration of the training
                  epochs=50,  # no of iternation
                  dropouts = 0.1, # Express the percentage of nodes that
                  # will be randomly dropped as a decimal.
                  probability = 1,
                  verbose=2)
                  
    
    
    clf.fit(embeddings, labelsNum)

    fName = "./models/classifier.pkl"
    print("Saving classifier")
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true')

    parser.add_argument(
        '--classifier',
        type=str,
        choices=[
            'LinearSvm',
            'GridSearchSvm',
            'GMM',
            'RadialSvm',
            'DecisionTree',
            'GaussianNB',
            'DBN'],
        help='The type of classifier to use.',
        default='RadialSvm')

    args = parser.parse_args()    
    
    train()




