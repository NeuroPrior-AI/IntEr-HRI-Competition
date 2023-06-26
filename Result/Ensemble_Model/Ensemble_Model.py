import os
import numpy as np
import pickle
import itertools
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Conv1D, Flatten, Dense, LSTM
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from joblib import dump
import seaborn as sns
from sklearn.model_selection import cross_val_score

def create_lstm():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(chans, samples)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class KerasClassifierWithProba(KerasClassifier):
    def predict_proba(self, X):
        preds = self.model.predict(X)
        return preds  # Assuming softmax activation function in the output layer

    _estimator_type = "classifier"

def create_cnn():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(chans, samples)))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap='Blues'):
    plt.figure(figsize=(10,7))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(folder_path, 'confusion_matrix.png'))
    plt.show()

subject_path='/Users/zhezhengren/Desktop/NeuroPrior_AI/Model_Competion/EEG/Data_Processing/New/data_by_subject/AJ05D'

with open(subject_path + '/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open(subject_path + '/y.pkl', 'rb') as f:
    y = pickle.load(f)
y = np_utils.to_categorical(y-1)

chans, samples = 64, 501
n_components = 3

clfs = [('mlp', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, activation='relu', solver='adam', random_state=1))),
        ('logreg', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), LogisticRegression(solver='liblinear'))),
        ('svc', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), SVC(kernel='rbf', C=1, gamma=0.01, probability=True))),
        ('rf', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), RandomForestClassifier(n_estimators=50))),
        ('knn', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), KNeighborsClassifier(n_neighbors=5))),
        ('xgb', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), XGBClassifier()))]

cnn = KerasClassifierWithProba(build_fn=create_cnn, epochs=10, batch_size=10, verbose=0)

lstm = KerasClassifierWithProba(build_fn=create_lstm, epochs=10, batch_size=10, verbose=0)
clfs.append(('lstm', lstm))

parameters = {'n_estimators': [100, 150, 200]}
clf_xgb = GridSearchCV(XGBClassifier(), parameters)
clfs.append(('xgb_gs', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), clf_xgb)))

clfs.append(('cnn', cnn))

clf = VotingClassifier(estimators=clfs, voting='soft')

kf = KFold(n_splits=10, shuffle=True, random_state=42)
validation_scores = []
class_names = ['no error', 'S 96', 'S 80']

i = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_flattened, Y_train.argmax(axis=-1))
    X_res = X_res.reshape(X_res.shape[0], chans, samples)

    clf.fit(X_res, y_res)
    preds_rg = clf.predict(X_test)

    acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
    f1 = f1_score(Y_test.argmax(axis=-1), preds_rg, average='macro')
    validation_scores.append(acc2)

    folder_path = os.path.join(subject_path, f"Validation_{i}")
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, 'accuracy.txt'), 'w') as f:
        f.write(f"Classification accuracy: {acc2*100:.4f}%")  # As percentage
    with open(os.path.join(folder_path, 'f1_score.txt'), 'w') as f:
        f.write(f"F1 score: {f1*100:.4f}%")  # As percentage

    cm = confusion_matrix(Y_test.argmax(axis=-1), preds_rg)
    plot_confusion_matrix(cm, class_names, normalize=True, title='Normalized confusion matrix')

    dump(clf, os.path.join(folder_path, 'Ensemble.joblib'))

    i += 1

#cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#cv_results = cross_val_score(clf, X, y.argmax(axis=-1), cv=cv)
#cv_accuracy = np.mean(cv_results)

# Save 10-fold cross validation accuracy
#with open(os.path.join(subject_path, 'cv_accuracy.txt'), 'w') as f:
#    f.write("10-fold cross validation accuracy: " + str(cv_accuracy))
#
#print("10-fold cross validation accuracy: ", cv_accuracy)
