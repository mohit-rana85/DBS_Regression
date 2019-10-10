import os
import yaml
import pickle as pkl
from tkinter import filedialog
from tkinter import *
from src.feature_select import FeatureSelect
from src.import_data import ImportData
from src.regression_ana import RegressionAnalysis


def dbs_regression():
    print("Select a data folder")
    root = Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(initialdir='/media/rana/Data/Project_data/Dystonia_Mayo/')
    if len(data_dir) == 0:
        print("Please select the data directory!!")
        quit()
    print('Selected data directory:')
    print(data_dir)
    config_files = [f1 for f1 in os.listdir(os.path.join(os.getcwd(), 'config')) if f1.endswith('yaml')]
    for config in config_files:
        config_file = open(os.path.join(os.getcwd(), 'config', config), 'r')
        config_para = yaml.load(config_file, Loader=yaml.Loader)
        output_dir = os.path.join(os.getcwd(), 'output', config[:-5])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Importing data files
        if not os.path.exists(os.path.join(output_dir, "imported_data" + ".pkl")):
            imported_data = ImportData(data_dir, config_para,  output_dir)
            df = imported_data.extract_mask_data()
            imported_data.extract_brain_data(df=df)
        else:
            print('Using existing Imported data')
        print("  ")
        print("Selecting features from the data")
        if not os.path.exists(os.path.join(output_dir, "features_data" + ".pkl")):
             features = FeatureSelect(os.path.join(output_dir, "imported_data" + ".pkl"), output_dir, config_para)
             features.feature_select()
        else:
            print('Using computed feature data.')
        print("computing grid regression analysis and plotting data using MI feature selection method ")
        reg_ana = RegressionAnalysis(config_para)
        reg_ana.regression_ana(output_dir)


if __name__ == '__main__':
    dbs_regression()
    print('Finished Analysis!')
