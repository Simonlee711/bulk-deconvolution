# SVR

#%%
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

# stats & plot specific from .py files
from src.plot import Plot
from stats import statsTest
from src.helper import flatten, gene_intersection
import src.project_configs as project_configs
from src.deconv import Deconvolution

from tqdm import tqdm

#%%

signature = pd.read_csv('../data/ibd_signature_whole.csv',index_col=0).T
atap_bulk = pd.read_csv('../data/atap.pseudobulk_counts.log_normal.csv',index_col=0).T
atap_true = pd.read_csv('../data/atap.scg_proportions.csv',index_col=0).T
cleveland_bulk = pd.read_csv('../data/cleveland.pseudobulk_counts.log_normal.csv',index_col=0).T
cleveland_true = pd.read_csv('../data/cleveland.scg_proportions.csv',index_col=0).T
lmu_bulk = pd.read_csv('../data/lmu.pseudobulk_counts.log_normal.csv',index_col=0).T
lmu_true = pd.read_csv('../data/lmu.scg_proportions.csv',index_col=0).T

training_data = atap_bulk
#%%
set1 = set(training_data.index)
set2 = set(signature.index)
intersection = set1.intersection(set2)
inter = list(intersection)

signature = signature.filter(items=inter, axis=0)
training_data = training_data.filter(items=inter, axis=0)

# transform data
ind = training_data.columns

train = signature.to_numpy()
test_data = training_data.to_numpy()
Nus = [0.25, 0.5, 0.75]

SVRcoef1 = np.zeros((signature.shape[1], training_data.shape[1]))
Selcoef1 = np.zeros((training_data.shape[0], training_data.shape[1]))

for i in tqdm(range(training_data.shape[1])):
    sols = [NuSVR(kernel="linear", nu=nu).fit(train, test_data[:, i]) for nu in Nus]
    im_name = signature.columns
    RMSE = [mse(sol.predict(train), test_data[:, i]) for sol in sols]
    Selcoef1[sols[np.argmin(RMSE)].support_, i] = 1
    SVRcoef1[:, i] = np.maximum(sols[np.argmin(RMSE)].coef_, 0)
    SVRcoef1[:, i] = SVRcoef1[:, i] / np.sum(SVRcoef1[:, i])
svr_1 = pd.DataFrame(SVRcoef1, index=im_name, columns=ind)
svr_1 = svr_1.reindex(sorted(svr_1.columns), axis=1)

# %%
display(svr_1)
#%%
cellanneal_pred.to_csv("../data/results2/cellann.csv")
# %%
