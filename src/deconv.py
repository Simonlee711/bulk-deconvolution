'''
Deconvolution Model Wrapper
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random

# import deconvolution methods
import cellanneal
from Kassandra.core.mix import Mixers
from Kassandra.core.cell_types import CellTypes
from Kassandra.core.model import DeconvolutionModel
from Kassandra.core.plotting import print_cell_matras, cells_p, print_all_cells_in_one
from Kassandra.core.utils import *
from scipy.optimize import nnls
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler

# statistical tests
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from tqdm import tqdm

class Deconvolution():

    def train(signature, bulk):
        '''
        Loads in the methods and retrains from scratch

        Parameters:

        Returns:
        '''

        # cell anneal
        gene_dict = cellanneal.make_gene_dictionary(
            signature,
            bulk,
            disp_min=0.5,
            bulk_min=1e-5,
            bulk_max=0.01)
        

        # Kassandra
        cell_types = CellTypes.load('./Kassandra/configs/custom.yaml')
        mixer = Mixers(cell_types=cell_types,
                    cells_expr=bulk, cells_annot=signature,
                    tumor_expr=bulk, tumor_annot=signature,
                    num_av=3, num_points=3000)
        model = DeconvolutionModel(cell_types,
                                boosting_params_first_step='./Kassandra/configs/boosting_params/median_model_first.tsv',
                                boosting_params_second_step='./Kassandra/configs/boosting_params/median_model_second.tsv')
        model.fit(mixer)


        # SVR
        ind = bulk.columns
        Nus=[0.25, 0.5, 0.75]

        SVRcoef = np.zeros((signature.shape[1], bulk.shape[1]))
        Selcoef = np.zeros((bulk.shape[0], bulk.shape[1]))

        train = signature
        test_data = bulk
        for i in tqdm(range(bulk.shape[1])):
            sols = [NuSVR(kernel='linear', nu=nu).fit(train,test_data[:,i]) for nu in Nus]
            im_name = signature.columns
            RMSE = [mse(sol.predict(train), test_data[:,i]) for sol in sols]
            Selcoef[sols[np.argmin(RMSE)].support_, i] = 1
            SVRcoef[:,i] = np.maximum(sols[np.argmin(RMSE)].coef_,0)
            SVRcoef[:,i] = SVRcoef[:,i]/np.sum(SVRcoef[:,i])
        svr_preds = pd.DataFrame(SVRcoef,index=im_name, columns=ind)
        svr_preds = svr_preds.reindex(sorted(svr_preds.columns), axis=1)
        svr_preds

        return gene_dict, model, svr_preds

    def deconvolution(models, test_data):
        '''
        Pass in models from training along with testing set to proceed with benchmarking

        Parameters:
            models: a list of models that contain the different saved methods
            test_data: the dataset being tested on
        
        Returns:
            returns a prediction or a list of prediction matrices needed for benchmarking
        '''


    
        models = models
        test_data = test_data

    