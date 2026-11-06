from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

rkf_repeats = 10
mc_test_size = 0.2
mc_n_splits_five = 5
mc_n_splits_twenty = 20
mc_n_splits_fifty = 50

def execute_kf_cv(X, y, num_iterations):
    mse_kfold_cv = []
    for i in range(num_iterations):
        kf = KFold(n_splits=5, shuffle=True, random_state=i)
        mse_kfold = []
        for train_index, test_index in kf.split(X):
            X_train_kf, X_test_kf = X[train_index], X[test_index]
            y_train_kf, y_test_kf = y[train_index], y[test_index]
            knr = KNeighborsRegressor(n_neighbors=3).fit(X_train_kf, y_train_kf)
            y_pred_kf = knr.predict(X_test_kf)
            mse_kfold.append(mean_squared_error(y_test_kf, y_pred_kf))
        mse_kfold_cv.append(np.mean(mse_kfold))
    return mse_kfold_cv

def execute_mc_cv(X, y, num_iterations, n_splits):
    #mc_test_size = 0.2
    mse_mc_cv = []
    for i in range(num_iterations):
        ss = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=i)
        mse_mc = []
        for train_index, test_index in ss.split(X):
            X_train_mc, X_test_mc = X[train_index], X[test_index]
            y_train_mc, y_test_mc = y[train_index], y[test_index]
            knr = KNeighborsRegressor(n_neighbors=3).fit(X_train_mc, y_train_mc)
            y_pred_mc = knr.predict(X_test_mc)
            mse_mc.append(mean_squared_error(y_test_mc, y_pred_mc))
        mse_mc_cv.append(np.mean(mse_mc))
    return mse_mc_cv

def execute_rkf_cv(X, y, num_iterations, rkf_repeats):
    mse_rkfold_cv = []
    for i in range(num_iterations):
        # Repeated K-Fold Cross-Validation
        rkf = RepeatedKFold(n_splits=5, n_repeats=rkf_repeats, random_state=i)
        mse_rkf = []
        for train_index, test_index in rkf.split(X):
            X_train_rkf, X_test_rkf = X[train_index], X[test_index]
            y_train_rkf, y_test_rkf = y[train_index], y[test_index]
            knr = KNeighborsRegressor(n_neighbors=3).fit(X_train_rkf, y_train_rkf)
            y_pred_rkf = knr.predict(X_test_rkf)
            mse_rkf.append(mean_squared_error(y_test_rkf, y_pred_rkf))
        mse_rkfold_cv.append(np.mean(mse_rkf))
    return mse_rkfold_cv
