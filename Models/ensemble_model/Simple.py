import numpy as np
import pickle
from tensorflow.keras import utils as np_utils
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
# from autosklearn.classification import AutoSklearnClassifier
from joblib import dump

n_class = 2
class KerasClassifierWithProba(KerasClassifier):
    def predict_proba(self, X):
        preds = self.model.predict(X)
        return preds  # Assuming softmax activation function in the output layer

    _estimator_type = "classifier"


def create_lstm():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(chans, samples)))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def create_cnn():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=n_class,
              activation='relu', input_shape=(chans, samples)))
    model.add(Flatten())
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


all_data_path = '../../Preprocess'
with open(all_data_path + '/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open(all_data_path + '/y.pkl', 'rb') as f:
    y = pickle.load(f)
y = np_utils.to_categorical(y-1)

print("Original shape of X:", X.shape)

# Split into 70% training, 10% validation, 20% testing:
X_train, X_temp, Y_train, Y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.66667)

chans, samples = 64, 501
n_components = 3

# train deep learning models
cnn = KerasClassifierWithProba(
    build_fn=create_cnn, epochs=10, batch_size=10, verbose=0)
lstm = KerasClassifierWithProba(
    build_fn=create_lstm, epochs=10, batch_size=10, verbose=0)

# train AutoML model
# automl = AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
# automl.fit(X, y.argmax(axis=-1))

# Add PCA
pca = PCA(n_components=n_components)
# reshape X to two dimensions
X_2d = X.reshape(X.shape[0], -1)
X_pca = pca.fit_transform(X_2d)

clfs = [
    ('mlp', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), MLPClassifier(
        hidden_layer_sizes=(50,), max_iter=1000, activation='relu', solver='adam', random_state=1))),
    ('logreg', make_pipeline(XdawnCovariances(n_components), TangentSpace(
        metric='riemann'), LogisticRegression(solver='liblinear'))),
    ('svc', make_pipeline(XdawnCovariances(n_components), TangentSpace(
        metric='riemann'), SVC(kernel='rbf', C=1, gamma=0.01, probability=True))),
    ('rf', make_pipeline(XdawnCovariances(n_components), TangentSpace(
        metric='riemann'), RandomForestClassifier(n_estimators=50))),
    ('knn', make_pipeline(XdawnCovariances(n_components), TangentSpace(
        metric='riemann'), KNeighborsClassifier(n_neighbors=5))),
    ('xgb', make_pipeline(XdawnCovariances(n_components),
     TangentSpace(metric='riemann'), XGBClassifier())),
    ('gb', make_pipeline(XdawnCovariances(n_components),
     TangentSpace(metric='riemann'), GradientBoostingClassifier())),
    ('catboost', make_pipeline(XdawnCovariances(n_components),
     TangentSpace(metric='riemann'), CatBoostClassifier(verbose=0))),
    ('lstm', lstm),
    ('cnn', cnn)
]

parameters = {'n_estimators': [100, 150, 200]}
clf_xgb = GridSearchCV(XGBClassifier(), parameters)
clfs.append(('xgb_gs', make_pipeline(XdawnCovariances(
    n_components), TangentSpace(metric='riemann'), clf_xgb)))

clf = VotingClassifier(estimators=clfs, voting='soft')

X_flattened = X.reshape(X.shape[0], -1)
ad = ADASYN(random_state=42)

X_res, y_res = ad.fit_resample(X_flattened, y.argmax(axis=-1))
print("Shape of X_res after resampling:", X_res.shape)


X_res = X_res.reshape(X_res.shape[0], chans, samples)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
validation_scores = []
class_names = ['non-P300', 'P300']

clf.fit(X_res, y_res)

dump(clf, 'Ensemble.joblib')
preds_rg = clf.predict(X_test)

acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc2))

f1 = f1_score(y.argmax(axis=-1), preds_rg, average='weighted')
print("F1 Score: %f" % f1)

plot_confusion_matrix(preds_rg, y.argmax(axis=-1),
                      class_names, title='Ensemble Method')
plt.savefig('cm_rg.png')
