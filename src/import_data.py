import os
import nibabel as nib
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import minmax_scale


class ImportData:
    def __init__(self, data_dir, config_para,  output_dir):
        self.data_dir = data_dir
        self.beh_data_file = config_para['beh_file']
        self.data_column = config_para['data_columns']
        self.fmri_file = config_para['fmri_file']
        self.dmri_file = config_para['dmri_file']
        self.file_ext = config_para['img_ext']
        self.roi_root_path = config_para['roi_root_path']
        self.roi_paths = config_para['roi_paths']
        self.data_method = config_para['data_method']
        self.mask_names = np.array([])
        self.roi_num = 0
        self.threshold_feature = config_para['threshold']
        self.output_dir = output_dir

    def extract_mask_data(self):
        # importing the Behavioral scores
        mask_file = dict()
        mask_data = dict()
        print('Import behavioral data from : {}'.format(self.beh_data_file))
        data_excel = pd.read_excel(os.path.join(self.data_dir, self.beh_data_file))
        df = pd.DataFrame(data_excel, columns=self.data_column)
        f = open(os.path.join(self.output_dir, 'beh_data.pkl'), 'wb')
        pkl.dump(df, f)
        f.close()
        row_dir = 0

        # loading the mask files
        for roi_path in self.roi_paths:
            mask_files = [f1 for f1 in os.listdir(os.path.join(self.roi_root_path, roi_path)) if
                          f1.endswith(self.file_ext)]
            self.roi_num = len(mask_files)
            ii = 0
            print('Importing Mask for DTI')
            print('Number of Mask files: {}'.format(len(mask_files)))
            mask_data_complete = pd.DataFrame()
            # 1mm mask files
            for mask in mask_files:

                img = nib.load(os.path.join(self.roi_root_path, roi_path, mask))
                mask_names = np.append(self.mask_names, mask[:-4])
                if ii == 0:
                    hdr = img.header
                    img_size = hdr.get_data_shape()
                    data = img.get_data()
                    mask_data = np.zeros(shape=(img_size[0] * img_size[1] * img_size[2], 1), dtype=int)
                    mask_data[np.nonzero(data.flatten('C')), ii] = 1
                else:
                    mask_data = np.zeros(shape=(img_size[0] * img_size[1] * img_size[2], 1), dtype=int)
                    data = img.get_data()
                    mask_data[np.nonzero(data.flatten('C'))] = 1
                if len(mask_data_complete.columns) == 0:
                    mask_data_complete = pd.DataFrame(mask_data, columns=[mask[:-4]])
                else:
                    mask_data_complete[mask[:-4]] = mask_data
                del img, data, mask_data
                ii += 1
            f = open(os.path.join(self.output_dir, 'mask_data_{}.pkl'.format(self.data_method[row_dir])), 'wb')
            pkl.dump(mask_data_complete, f, protocol=4)
            f.close()
            row_dir += 1
            del mask_files, mask_data_complete
        return df

    def extract_brain_data(self, df):
        # loading the data files for each subject
        complete_data = np.array([])
        filenames = [f1 for f1 in os.listdir(self.output_dir) if f1.endswith('.pkl')]
        if self.data_method[0] == 'fmri':
           nn = [idx for idx, s in enumerate(filenames) if 'mask_data_{}'.format(self.data_method[0]) in s]
        else:
           nn = [idx for idx, s in enumerate(filenames) if 'mask_data_{}'.format(self.data_method[1]) in s]
        f = open(os.path.join(self.output_dir, filenames[nn[0]]), 'rb')
        pd_mask_data_fmri = pkl.load(f)
        f.close()
        del f, nn
        if self.data_method[0] == 'dmri':
            nn = [idx for idx, s in enumerate(filenames) if 'mask_data_{}'.format(self.data_method[0]) in s]
        else:
            nn = [idx for idx, s in enumerate(filenames) if 'mask_data_{}'.format(self.data_method[1]) in s]
        f = open(os.path.join(self.output_dir, filenames[nn[0]]), 'rb')
        pd_mask_data_dmri = pkl.load(f)
        f.close()
        del f, nn
        subj_num = -1
        dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        score = np.zeros(shape=[1, len(self.data_column)-1], dtype=float)
        for d in dirs:
            subj_num += 1
            xx = d.isdigit()
            if xx == 1:
                data_dmri = np.array([])
                data_fmri = np.array([])
                print("Extracting the data for each roi ")
                idx = df.loc[df['INFORM_ID'] == int(d)].index.values
                score = np.append(score, df.iloc[idx[0], :].to_numpy()[1:].reshape(1, -1), axis=0)
                filenames = [f1 for f1 in os.listdir(os.path.join(self.data_dir, d)) if f1.endswith(self.file_ext)]
                print("Subject: " + d)
                index = [idx for idx, s in enumerate(filenames) if self.dmri_file in s][0]
                print('dMRI')
                img = nib.load(os.path.join(self.data_dir, d, filenames[index]))
                data_1mm = minmax_scale(np.array(img.get_data().flatten('C')).T, [-1, 1])
                for ind, column in enumerate(pd_mask_data_dmri.columns):
                    data_com = np.extract(pd_mask_data_dmri[column].to_numpy() == 1, data_1mm)
                    low_per = np.percentile(data_com, self.threshold_feature[0])
                    high_per = np.percentile(data_com, self.threshold_feature[1])
                    data_dmri = np.append(data_dmri, np.mean(data_com[np.where(data_com <= low_per)], axis=0))
                    data_dmri = np.append(data_dmri, np.mean(data_com[np.logical_and(data_com > low_per,
                                                                                     data_com > high_per)], axis=0))
                    data_dmri = np.append(data_dmri, np.mean(data_com[np.where(data_com >= high_per)], axis=0))

                    del low_per, high_per
                index = [idx for idx, s in enumerate(filenames) if self.fmri_file in s][0]
                del data_1mm
                print('fMRI')
                img = nib.load(os.path.join(self.data_dir, d, filenames[index]))
                data_2mm = minmax_scale(np.array(img.get_data().flatten('C')).T, [0, 1])
                for ind, column in enumerate(pd_mask_data_fmri.columns):
                    data_com = np.extract(pd_mask_data_fmri[column].to_numpy() == 1, data_2mm)
                    low_per = np.percentile(data_com, self.threshold_feature[0])
                    high_per = np.percentile(data_com, self.threshold_feature[1])
                    data_fmri = np.append(data_fmri, np.mean(data_com[np.where(data_com <= low_per)], axis=0))
                    data_dmri = np.append(data_dmri, np.mean(data_com[np.logical_and(data_com > low_per,
                                                                                     data_com > high_per)], axis=0))
                    data_fmri = np.append(data_fmri, np.mean(data_com[np.where(data_com >= high_per)], axis=0))
                data_dmri = np.concatenate((data_dmri, data_fmri), axis=0)
                del data_2mm
                if subj_num == 0:
                    complete_data = data_dmri
                else:
                    complete_data = np.vstack((complete_data, data_dmri))
                data_dmri = np.array([])
                data_fmri = np.array([])
        score = np.delete(score, 0, axis=0)
        f = open(os.path.join(self.output_dir, "imported_data" + ".pkl"), 'wb')
        imported_data = (complete_data, score, subj_num, self.roi_num, self.mask_names)
        pkl.dump(imported_data, f)
        f.close()



