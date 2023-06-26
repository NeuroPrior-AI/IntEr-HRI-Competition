import numpy as np
import pickle
from tensorflow.keras import utils as np_utils

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances, ERPCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from pyriemann.classification import MDM
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_decomposition import CCA
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from joblib import dump

############################# Load data ##################################

# Load the data
subject = 'BS34D'
subject_path = '../Process/data_by_subject/' + subject
all_data_path = '../Process/'

with open(all_data_path + '/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open(all_data_path + '/y.pkl', 'rb') as f:
    y = pickle.load(f)

y = np_utils.to_categorical(y-1)

# take 50/25/25 percent of the data to train/validate/test
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, y, test_size=0.5, shuffle=True)
X_validate, X_test, Y_validate, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, shuffle=True)


############################# Set Parameters ##############################

chans, samples = X.shape[1], X.shape[2]

############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in
# the tangent space with a logistic regression

nfilter = 5  # pick some components
est, xest = "scm", "lwf"
# set up sklearn pipeline
# clf = make_pipeline(XdawnCovariances(
#                         nfilter=nfilter,
#                         applyfilters=True,
#                         estimator=est,
#                         xdawn_estimator=xest,
#                         baseline_cov=None
#                         ),
#                     TangentSpace(metric='riemann'),
#                     MLPClassifier((100,), learning_rate='adaptive', batch_size=64))
clf = make_pipeline(XdawnCovariances(nfilter),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())
# cov_est = XdawnCovariances(
#         nfilter=n_components,
#         # applyfilters=True,
#         # classes=[ep_calib.event_id["Target"]],
#         # estimator=est,
#         # xdawn_estimator=xest,
#         # baseline_cov=None,
#     )
# mdm = MDM(metric="riemann")
# clf = Pipeline([("cov_est", cov_est), ("mdm", mdm)])
preds_rg = np.zeros(len(Y_test))

# reshape back to (trials, channels, samples)
X_train = X_train.reshape(X_train.shape[0], chans, samples)
X_test = X_test.reshape(X_test.shape[0], chans, samples)

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
clf.fit(X_train, Y_train.argmax(axis=-1))
dump(clf, 'xDAWN.joblib')
preds_rg = clf.predict(X_test)

# Printing the results
acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc2))


############################# Confusion Matrix Portion ##############################
names = ['no error', 'error', 'reaction']

plt.figure(0)
plot_confusion_matrix(preds_rg, Y_test.argmax(
    axis=-1), names, title='xDAWN + RG')
plt.savefig('cm_rg.png')


############################# 10-fold Cross Validation ##############################
cv_results = cross_validate(clf, X, y.argmax(axis=-1), cv=10)
print("10-fold cross validation accuracy: ", np.mean(cv_results['test_score']))
