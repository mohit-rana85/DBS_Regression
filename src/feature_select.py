import pickle as pkl
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import os


class FeatureSelect:
    def __init__(self, data_file, output_dir, config_para):
        self.data_file = data_file
        self.output_dir = output_dir
        self.feature_val = np.array([])
        self.columns_len = len(config_para['data_columns']) - 1

    def feature_select(self):
        with open(os.path.join(self.data_file), 'rb') as f:
            imported_data = pkl.load(f)
        complete_data, score, subj_num, self.roi_num, self.mask_names = imported_data
        del imported_data
        print("Computing features")
        print()
        print('Computing features using Mututal information and Correlation Coefficients.')
        corr = np.zeros(shape=[self.columns_len, complete_data.shape[1]], dtype=float)
        mi = np.zeros(shape=[self.columns_len, complete_data.shape[1]], dtype=float)
        for ii in range(self.columns_len):
            mi[ii, :] = mutual_info_regression(complete_data, score[:, ii])
            for iin in range(complete_data.shape[1]):
                corr1 = np.corrcoef(complete_data[:, iin], score[:, ii])
                corr[ii, iin] = corr1[0, 1]
                del corr1
        self.feature_val = np.vstack((mi, corr))
        f = open(os.path.join(self.output_dir, "features_data" + ".pkl"), 'wb')
        pkl.dump(self.feature_val, f)
        f.close()









