from scipy.stats import stats
from sklearn import linear_model
from sklearn import ensemble
import sklearn.svm as svm
import sklearn.metrics as skm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os


class RegressionAnalysis:
    def __init__(self, config_para):
        # Dictionary of pipelines and classifier types for ease of reference
        self.classifiers_name = {0: 'Support Vector Machine Regression', 1: 'Stochastic Gradient Descent' ,
                                 2: 'Bayesian Regression', 3: 'LassoLars', 4: 'Lasso', 5: 'ARD', 6: 'Passive Aggressive'
                                 , 7: 'TheilSen', 8: 'Linear', 9: 'Ridge', 10: 'ElasticNet', 11: 'RandomForest'}
        self.classifiers = [svm.SVR(), linear_model.SGDRegressor(), linear_model.BayesianRidge(),
                            linear_model.LassoLars(), linear_model.Lasso(), linear_model.ARDRegression(),
                            linear_model.PassiveAggressiveRegressor(), linear_model.TheilSenRegressor(),
                            linear_model.LinearRegression(), linear_model.Ridge(), linear_model.ElasticNet(),
                            ensemble.RandomForestRegressor()]
        self.n_jobs = config_para['n_jobs']
        self.cv = config_para['cv']
        self.cond_num = len(config_para['data_columns'])-1
        self.num_features = config_para['num_feature']
        param_range = [10, 50, 100, 300, 500, 700, 1000]
        param_range_fl = [1.0, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001, 0.0001]
        # Linear regression
        grid_params_lr = [{'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}]
        # SVR
        grid_params_svm = [{'C': param_range, 'kernel': ['linear']},
                           {'C': param_range, 'gamma': param_range_fl, 'kernel': ['rbf']}]
        # SGD
        grid_params_sgd = [{'alpha': param_range_fl,
                            'penalty': ['l1', 'l2', 'elasticnet'],
                            'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']}]
        # BayesianRidge
        grid_params_br = [{'alpha_1': param_range_fl,
                           'alpha_2': param_range_fl,
                           'tol': [0.0001, 0.001, 0.01, 0.1],
                           'n_iter': [1000, 2000, 5000]}]
        # LassoLars
        grid_params_ll = [{'alpha': param_range_fl,
                           'eps': [1e-12, 1e-15, 1e-16],
                           'max_iter': [1000, 2000, 5000]}]
        # Lasso
        grid_params_lasso = [{'alpha': param_range_fl,
                              'tol': [0.0001, 0.001, 0.01, 0.1],
                              'max_iter': [1000, 2000, 5000]}]
        # ARD
        grid_params_ard = [{'alpha_1': param_range_fl,
                            'alpha_2': param_range_fl,
                            'tol': [0.0001, 0.001, 0.01, 0.1],
                            'max_iter': [1000, 2000, 5000]}]
        # PassiveAggressive
        grid_params_pa = [{'C': param_range,
                           'max_iter': [1000, 2000, 5000]}]
        # TheilSen
        grid_params_ts = [{'max_iter': [1000, 2000, 5000],
                           'tol': [0.0001, 0.001, 0.01, 0.1]}]

        # Ridge
        grid_params_ridge = [{'max_iter': [1000, 2000, 5000],
                              'tol': [0.0001, 0.001, 0.01, 0.1],
                              'alpha': param_range_fl}]
        # ElasticNet
        grid_params_en = [{'max_iter': [1000, 2000, 5000],
                           'tol': [0.0001, 0.001, 0.01, 0.1],
                           'alpha': param_range_fl}]
        # RandomForest
        grid_params_rf = [{'criterion': ['mse', 'mae'],
                           'max_depth': range(1, 1000, 50),
                           'n_estimators': range(1, 1000, 50)}]
        gs_lr = GridSearchCV(estimator=linear_model.LinearRegression(),
                             param_grid=grid_params_lr, scoring='neg_mean_squared_error',
                             cv=self.cv, n_jobs=self.n_jobs)
        gs_svm = GridSearchCV(estimator=svm.SVR(),
                              param_grid=grid_params_svm, scoring='neg_mean_squared_error',
                              cv=self.cv, n_jobs=self.n_jobs)
        gs_sgd = GridSearchCV(estimator=linear_model.SGDRegressor(),
                              param_grid=grid_params_sgd, scoring='neg_mean_squared_error',
                              cv=self.cv, n_jobs=self.n_jobs)
        gs_br = GridSearchCV(estimator=linear_model.BayesianRidge(),
                             param_grid=grid_params_br, scoring='neg_mean_squared_error',
                             cv=self.cv, n_jobs=self.n_jobs)
        gs_en = GridSearchCV(estimator=linear_model.ElasticNet(),
                             param_grid=grid_params_en, scoring='neg_mean_squared_error',
                             cv=self.cv, n_jobs=self.n_jobs)
        gs_rf = GridSearchCV(estimator=ensemble.RandomForestRegressor(),
                             param_grid=grid_params_rf, scoring='neg_mean_squared_error',
                             cv=self.cv, n_jobs=self.n_jobs)
        gs_ridge = GridSearchCV(estimator=linear_model.Ridge(),
                                param_grid=grid_params_ridge, scoring='neg_mean_squared_error',
                                cv=self.cv, n_jobs=self.n_jobs)
        gs_ts = GridSearchCV(estimator=linear_model.TheilSenRegressor(),
                             param_grid=grid_params_ts, scoring='neg_mean_squared_error',
                             cv=self.cv, n_jobs=self.n_jobs)
        gs_pa = GridSearchCV(estimator=linear_model.PassiveAggressiveRegressor(),
                             param_grid=grid_params_pa, scoring='neg_mean_squared_error',
                             cv=self.cv, n_jobs=self.n_jobs)
        gs_ard = GridSearchCV(estimator=linear_model.ARDRegression(),
                              param_grid=grid_params_ard, scoring='neg_mean_squared_error',
                              cv=self.cv, n_jobs=self.n_jobs)
        gs_ll = GridSearchCV(estimator=linear_model.LassoLars(),
                             param_grid=grid_params_ll, scoring='neg_mean_squared_error',
                             cv=self.cv, n_jobs=self.n_jobs)
        gs_lasso = GridSearchCV(estimator=linear_model.Lasso(),
                                param_grid=grid_params_lasso, scoring='neg_mean_squared_error',
                                cv=self.cv, n_jobs=self.n_jobs)
        # List of pipelines for ease of iteration
        self.grids = [gs_lr, gs_svm, gs_sgd, gs_br, gs_en, gs_rf, gs_ridge, gs_ts, gs_pa, gs_ard, gs_ll, gs_lasso]

    def regression_ana(self, output_dir):
        f = open(os.path.join(output_dir, "feature_data_file" + ".pkl"), 'rb')
        feature_val = pkl.load(f)
        f.close()
        del f
        f = open(os.path.join(output_dir, "imported_data" + ".pkl"), 'rb')
        imported_data = pkl.load(f)
        complete_data, score, subj_num, roi_num, mask_names = imported_data
        f.close()
        del imported_data
        model_rms = np.zeros(shape=[6, len(self.classifiers_name)], dtype=float)
        model_rsq = np.zeros(shape=[6, len(self.classifiers_name)], dtype=float)
        xx = np.arange(subj_num)
        loo = LeaveOneOut()
        for num_feature in self.num_features:
            idx = -1
            cond_num = -1
            fig = plt.figure(figsize=[10, 8])
            for data_idx in range(3):  # Data: 0:DTI, 1:fMRI 2: DTI and fMRI
                if data_idx == 0:
                    print('Data to be analyzed: DTI')
                    data = complete_data[:, :3*roi_num]
                elif data_idx == 1:
                    print('Data to be analyzed: fMRI')
                    data = complete_data[:, 3 * roi_num:]
                elif data_idx ==2:
                    print('Data to be analyzed: DTI and fMRI')
                    data = complete_data
                for cond_idx in range(self.cond_num): # Conditions of columns
                    cond_num += 1
                    filename = os.path.join(output_dir, 'regression_plots_cond_' + str(cond_num) + "_num_fea_" +
                                            str(num_feature)+ ".png")
                    if data_idx == 1:
                        sorted_idx = np.argsort(feature_val[cond_idx, :3*roi_num])
                    elif data_idx == 1:
                        sorted_idx = np.argsort(feature_val[cond_idx, 3 * roi_num:])
                    elif data_idx == 2:
                        sorted_idx = np.argsort(feature_val[cond_idx, 3 * roi_num:])
                    training_data = data[:, sorted_idx[:num_feature]]
                    train_label = score[:, cond_idx]
                    for item in self.classifiers:
                        idx += 1
                        print(self.classifiers_name[idx])
                        clf = item
                        for train_idx, test_idx in loo.split(range(subj_num)):
                            print("Estimating the model")
                            clf.fit(training_data[train_idx], train_label[train_idx].ravel())
                            print(clf.best_index_)
                            print("Grid scores on development set:\n")
                            for params, mean_score, scores in clf.grid_scores_:
                                print("%0.3f (+/-%0.03f) for %r\n" % (mean_score, scores.std() * 2, params))
                            # Best params
                            print('Best params: %s' % clf.best_params_)
                            model_predict = np.append(model_predict, clf.predict(training_data[test_idx].reshape
                                                                                 (1, -1)))
                model_rms[cond_num, idx] = np.sqrt(skm.mean_squared_error(train_label, model_predict))
                model_rsq[cond_num, idx] = skm.r2_score(train_label, model_predict)
                self.regression_plot(train_label, model_predict, idx, self.model_rms[cond_num, idx],
                                     self.model_rsq[cond_num, idx], fig, filename)

    def regression_plot(self, data_label, model_predict, idx, rms, rsq, fig, fig_filename):
        # plotting predicted values
        fig.suptitle('Regression Analysis')
        axx = fig.add_subplot(3, 4, idx +1)
        plt.title(self.classifiers_name[idx] + ' RMS:' + str(np.round(rms)) + ' ( r2: ' +
                  str(np.round(rsq, 3)) + ')',
                  {'fontsize': 10, 'fontweight': 0.5, 'verticalalignment': 'baseline', 'horizontalalignment': 'center'})
        axx.scatter(data_label, model_predict, s=10, c='b', marker="s", label='Original')
        slope, intercept, r_value, p_value, std_err = stats.linregress(data_label, model_predict)

        line = slope * data_label + intercept
        plt.plot(data_label, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope, intercept))
        fig.legend('lower center')

        if idx == 11:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.show(block=False)
            plt.pause(3)
            print('saving file')
            plt.savefig(fig_filename, dpi=100)
            plt.close()


