#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:55:29 2023

@author: zhezhengren
"""

import numpy as np
import pickle
from tensorflow.keras import utils as np_utils
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt
from joblib import dump
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import LSTM
from sklearn.ensemble import StackingClassifier

n_class = 2


def create_lstm():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(chans, samples)))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


class KerasClassifierWithProba(KerasClassifier):
    def predict_proba(self, X):
        preds = self.model.predict(X)
        return preds  # Assuming softmax activation function in the output layer

    _estimator_type = "classifier"


def create_cnn():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=n_class,
              activation='relu', input_shape=(chans, samples)))
    model.add(Flatten())
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


#all_data_path = '../../Preprocess/subjects/BS34D'
all_data_path = '/home/naturaldx/IntEr-HRI-Competition/Preprocess/subjects/AQ59D'
with open(all_data_path + '/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open(all_data_path + '/y.pkl', 'rb') as f:
    y = pickle.load(f)
y = np_utils.to_categorical(y-1)

# X_train, X_temp, Y_train, Y_temp = train_test_split(X, y, test_size=0.5)
# X_validate, X_test, Y_validate, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5)

# Split into 70% training, 10% validation, 20% testing:
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.66667, random_state=42)

# X_train = X
# Y_train = y

chans, samples = 64, 501
n_components = 10
# Grid search for hyperparameter tuning
# clf_xgb = GridSearchCV(XGBClassifier(), param_grid={'n_estimators': [100, 150, 200]})

clfs = [
    ('mlp', make_pipeline(XdawnCovariances(n_components), TangentSpace(
        metric='riemann'), MLPClassifier((150,), learning_rate='adaptive', batch_size=64))),
    ('logreg', make_pipeline(XdawnCovariances(n_components),
     TangentSpace(metric='riemann'), LogisticRegression(C=1.0))),
    ('svm', make_pipeline(XdawnCovariances(n_components), TangentSpace(
        metric='riemann'), SVC(kernel='rbf', C=10, gamma='scale', probability=True))),
    ('rf', make_pipeline(XdawnCovariances(n_components), TangentSpace(
        metric='riemann'), RandomForestClassifier(n_estimators=150))),
    ('xgb', make_pipeline(XdawnCovariances(n_components), TangentSpace(
        metric='riemann'), XGBClassifier(n_estimators=150, learning_rate=0.1))),
    # ('knn', make_pipeline(XdawnCovariances(n_components), TangentSpace(
    #     metric='riemann'), KNeighborsClassifier(n_neighbors=5))),
    # ('xgb_gs', make_pipeline(XdawnCovariances(
    # n_components), TangentSpace(metric='riemann'), clf_xgb))
]

# cnn = KerasClassifierWithProba(build_fn=create_cnn, epochs=10, batch_size=10, verbose=0)
# lstm = KerasClassifierWithProba(build_fn=create_lstm, epochs=10, batch_size=10, verbose=0)
# clfs.append(('lstm', lstm))
# clfs.append(('cnn', cnn))

clf = VotingClassifier(estimators=clfs, voting='soft')
# clf = StackingClassifier(estimators=clfs, final_estimator=LogisticRegression(), stack_method='auto')

X_train_flattened = X_train.reshape(X_train.shape[0], -1)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_flattened, Y_train.argmax(axis=-1))

X_res = X_res.reshape(X_res.shape[0], chans, samples)

clf.fit(X_train, Y_train.argmax(axis=-1))
#dump(clf, '../pre-trained/Ensemble.joblib_AQ59D')
dump(clf, '/home/naturaldx/IntEr-HRI-Competition/Models/pre-trained/Ensemble_AQ59D.joblib')
X_test_flattened = X_test.reshape(X_test.shape[0], -1)
preds_rg = clf.predict(X_test)

acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc2))

names = ['no error', 'S 96']
plt.figure(0)
plot_confusion_matrix(preds_rg, Y_test.argmax(axis=-1), names, title='Ensemble Method')
#plt.savefig('../figures/ensemble-cv.png')
plt.savefig('/home/naturaldx/IntEr-HRI-Competition/Models/figures/ensemble-cv_AQ59D.png')

# cv_results = cross_validate(clf, X, y.argmax(axis=-1), cv=10)
# print("10-fold cross validation accuracy: ", np.mean(cv_results['test_score']))
# print("10-fold cross validation standard deviation: ", np.std(cv_results['test_score']))