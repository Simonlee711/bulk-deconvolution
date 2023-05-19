"""
Deconvolution Model Wrapper
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random

# import deconvolution methods
import cellanneal
# from Kassandra.core.mix import Mixers
# from Kassandra.core.cell_types import CellTypes
# from Kassandra.core.model import DeconvolutionModel
# from Kassandra.core.plotting import print_cell_matras, cells_p, print_all_cells_in_one
# from Kassandra.core.utils import *
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
from stats import statsTest


class Deconvolution:
    """
    A class for the deconvolution models
    """

    def deconvolution(self, signature, training_data):
        """
        Training models from scratch:
                      1. Cellanneal
                      2. Kassandra
                      3. SVR

        Parameters:
            These files are required for training
                      1. Gene Expression Signatures (pandas.Dataframe)
                      2. training dataset (list)
        Returns:
            The trained models are returned
                models (list): A list of trained models that will be returned:
                      1. Cellanneal model
                      2. Kassandra model
                      3. SVR
        """

        # cellanneal training

        # print("Preparing Cell Anneal Model")

        cellanneal_model = cellanneal.make_gene_dictionary(
            signature, training_data, disp_min=0.5, bulk_min=1e-5, bulk_max=0.01
        )
        
        cellanneal_pred = cellanneal.deconvolve(
            signature, training_data, maxiter=1000, gene_dict=cellanneal_model
        )
        # print("Training Kassandra Model")

        # # Kassandra training
        # lab_expr = pd.read_csv(
        #     "./Kassandra/trainingData/laboratory_data_expressions.tsv",
        #     sep="\t",
        #     index_col=0,
        # )
        # lab_annot = pd.read_csv(
        #     "./Kassandra/trainingData/laboratory_data_annotation.tsv",
        #     sep="\t",
        #     index_col=0,
        # )
        # lab_annot["Dataset"] = lab_annot.index
        # lab_annot = lab_annot.iloc[:, [1, 0]]

        # lab_expr

        # #change depending on tissue type
        # cell_types = CellTypes.load("./Kassandra/configs/custom.yaml")
        # mixer = Mixers(
        #     cell_types=cell_types,
        #     cells_expr=lab_expr,
        #     cells_annot=lab_annot,
        #     tumor_expr=lab_expr,
        #     tumor_annot=lab_annot,
        #     num_av=3,
        #     num_points=3000,
        # )


        # kassandra_model = DeconvolutionModel(
        #     cell_types,
        #     boosting_params_first_step="./Kassandra/configs/boosting_params/ibd_first_step.tsv",
        #     boosting_params_second_step="./Kassandra/configs/boosting_params/ibd_second_step.tsv",
        # )
        # kassandra_model.fit(mixer)

        ###################################################################################
        # pseudocode for SVR comes from https://rdrr.io/github/IOBR/IOBR/src/R/CIBERSORT.R
        ###################################################################################

        # print("Training SVR Model")

        # # get the gene intersection and filter them out
        # set1 = set(training_data.index)
        # set2 = set(signature.index)
        # intersection = set1.intersection(set2)
        # inter = list(intersection)

        # signature = signature.filter(items=inter, axis=0)
        # training_data = training_data.filter(items=inter, axis=0)

        # # transform data
        # ind = training_data.columns
        
        # train = signature.to_numpy()
        # test_data = training_data.to_numpy()
        # # else:
        # #     train = signature
        # #     test_data=training_data
        # Nus = [0.25, 0.5, 0.75]

        # SVRcoef1 = np.zeros((signature.shape[1], training_data.shape[1]))
        # Selcoef1 = np.zeros((training_data.shape[0], training_data.shape[1]))

        # for i in tqdm(range(training_data.shape[1])):
        #     sols = [NuSVR(kernel="linear", nu=nu).fit(train, test_data[:, i]) for nu in Nus]
        #     im_name = signature.columns
        #     RMSE = [mse(sol.predict(train), test_data[:, i]) for sol in sols]
        #     Selcoef1[sols[np.argmin(RMSE)].support_, i] = 1
        #     SVRcoef1[:, i] = np.maximum(sols[np.argmin(RMSE)].coef_, 0)
        #     SVRcoef1[:, i] = SVRcoef1[:, i] / np.sum(SVRcoef1[:, i])
        # svr_1 = pd.DataFrame(SVRcoef1, index=im_name, columns=ind)
        # svr_1 = svr_1.reindex(sorted(svr_1.columns), axis=1)

        models = cellanneal_pred

        return models

    # def deconvolution(self, models, test_data):
    #     """
    #     Pass in models from training along with testing set to proceed with benchmarking

    #     Parameters:
    #         models (list): a list of models that contain the different saved methods
    #                    1. Cellanneal
    #                    2. Kassandra
    #                    3. SVR

    #         test_data (list): the dataset being tested on in a particular order
    #                    1. Gene Expression Signature
    #                    2. bulk RNA-seq

    #     Returns:
    #         returns a prediction or a list of prediction matrices needed for benchmarking
    #                   1. cellanneal
    #                   2. Kassandra
    #                   3. SVR

    #     """

    #     # model extraction
    #     gene_dict = models[0]
    #     Kassandra = models[1]
    #     svr = models[2]

    #     # data extraction
    #     signature = test_data[0]
    #     gse1 = test_data[1]

    #     # for saving purposes
    #     prediction_saved_names = [
    #         "cellanneal.csv",
    #         "kassandra_all.csv",
    #         "svr.csv",
    #     ]
    #     i = 0

    #     #### Cell Anneal #####
    #     # GSE107572 bulk RNA-seq

    #     print("Performing cellanneal deconvolution \n")

    #     cellanneal_pred = cellanneal.deconvolve(
    #         signature, gse1, maxiter=1000, gene_dict=gene_dict
    #     )

    #     cellanneal_pred = cellanneal_pred * 100
    #     cellanneal_pred = cellanneal_pred.T.copy()

    #     ###################################################################################
    #     # !!! feel free to comment this out if working with a different tissue sample !!! #
    #     ###################################################################################

    #     # cellanneal_pred.loc["B_cells"] = cellanneal_pred.loc[["B", "B-naive"]].sum()
    #     # cellanneal_pred.loc["CD4_T_cells"] = cellanneal_pred.loc[
    #     #     ["CD4", "CD4-naive"]
    #     # ].sum()
    #     # cellanneal_pred.loc["CD8_T_cells"] = cellanneal_pred.loc[["CD8"]].sum()
    #     # cellanneal_pred.loc["NK_cells"] = cellanneal_pred.loc[["NK"]].sum()
    #     # cellanneal_pred.loc["Tregs"] = cellanneal_pred.loc[["Treg"]].sum()
    #     # cellanneal_pred.loc["T_cells"] = cellanneal_pred.loc[
    #     #     ["CD8_T_cells", "CD4_T_cells", "Tregs", "T_undef"]
    #     # ].sum()
    #     # cellanneal_pred.loc["Lymphocytes"] = cellanneal_pred.loc[
    #     #     ["B_cells", "T_cells", "NK_cells"]
    #     # ].sum()

    #     cellanneal_pred.to_csv(prediction_saved_names[i])
    #     i = i + 1

    #     #### Kassandra #####
    #     # GSE107572 bulk RNA-seq

    #     print("performing Kassandra deconvolution")

    #     kassandra_all = Kassandra.predict(gse1)
    #     ###################################################################################
    #     # !!! feel free to comment this out if working with a different tissue sample !!! #
    #     ###################################################################################
    #     # kassandra_all.loc["Lymphocytes"] = kassandra_all.loc[
    #     #     ["B_cells", "T_cells", "NK_cells"]
    #     # ].sum()

    #     kassandra_all = kassandra_all * 100
    #     ###
    #     kassandra_all.to_csv(prediction_saved_names[i])
    #     i = i + 1

    #     # # drop parent nodes so we can plot child nodes stack plots
    #     # parent_nodes = [
    #     #     "Non_plasma_B_cells",
    #     #     "Monocytes",
    #     #     "Granulocytes",
    #     #     "B_cells",
    #     #     "T_cells",
    #     #     "NK_cells",
    #     #     "Myeloid_cells",
    #     #     "Lymphoid_cells",
    #     #     "Lymphocytes",
    #     #     "CD8_T_cells",
    #     #     "Cytotoxic_NK_cells",
    #     #     "CD4_T_cells",
    #     #     "Memory_T_helpers",
    #     #     "Memory_CD8_T_cells",
    #     # ]
    #     # kassandra_child = kassandra_all.drop(parent_nodes)
    #     # ###
    #     # kassandra_child.to_csv(prediction_saved_names[i])
    #     # i = i + 1

    #     #### SVR ####
    #     # GSE107572 bulk RNA-seq

    #     print("Performing SVR deconvolution \n ")
    #     svr.to_csv(prediction_saved_names[i])
    #     i = i + 1

    #     # puts all dataframe objects into a list
    #     predictions = [cellanneal_pred, kassandra_all, svr]

    #     return predictions

    def benchmark(
        self,
        df_list,
        true,
        name_list=["cellanneal", "Kassandra", "SVR"],
        cell=True,
        sample=False,
        statistic="rmse",
    ):
        """
        A function that takes in all the predictions dataframes and benchmarks with true data measurements

        Parameters:
            df_list (list): a list object containing dataframes. Preferred order
                1. Cellanneal
                2. Kassandra
                3. SVR
            true (pandas.Dataframe): the true measurement to calculate the residuals at each cell level
            cell (bool): a boolean parameter to benchmark at the cell level (default: True)
            sample (bool): a boolean parameter to benchmark at sample level (default: False)
            statistic (string): choose one statistic to benchmark on (pearson, r2, diff, rmse)

        Returns:
            None
        """
        # first checks for whether only one of the flags are on
        # assert cell == True and sample == True, "Can't do that silly goose"

        # define some dummy variables for later
        df_final = pd.DataFrame()
        length = 999

        # Prepares the cell or sample specific dataframes
        preds_list = []
        for pred in df_list:
            if sample:
                pred = pred.T
            preds_list.append(pred)

        # now conducts the benchmarking
        for i, pred in enumerate(preds_list):
            ind_names = pred.dropna().index.intersection(true.dropna().index)
            col_names = pred.dropna().columns.intersection(true.dropna().columns)
            predicted_values = pred.loc[ind_names, col_names].astype(float)
            true_values = true.loc[ind_names, col_names].astype(float)
            print(predicted_values.shape)
            print(true_values.shape)
            predicted_values = predicted_values.T
            true_values = true_values.T
            cells = true_values.columns
            stat = statsTest()

            temp2 = predicted_values.shape[1]
            if temp2 < length:
                length = temp2

            # benchmarks based on statistic
            benchmark_list = []
            for x, cell in enumerate(cells):
                if statistic == "pearson":
                    val = stat.correlation(predicted_values[cell], true_values[cell])
                elif statistic == "r2":
                    val = stat.R_squared(predicted_values[cell], true_values[cell])
                elif statistic == "residual":
                    val =  true_values[cell] - predicted_values[cell]
                elif statistic == "rmse":
                    val = stat.rmse(predicted_values[cell], true_values[cell])
                else:
                    return "invalid prompt"
                if statistic == "residual":
                    benchmark_list.append(val[1])
                else:
                    benchmark_list.append(val)

            

            # create final dataframes
            benchmark_list = benchmark_list[0:length]
            df_final[name_list[i]] = benchmark_list
            if i == 0:
                index_name = predicted_values.columns
                df_final.index = index_name

        # how to score final to find minimums for rows
        if statistic == "pearson" or statistic == "r2":
            df_final["winners value"] = df_final[name_list].max(axis=1)
            df_final["winners"] = df_final[name_list].idxmax(axis=1)
        if statistic == "residual" or statistic == "rmse":
            df_final["winners value"] = df_final[name_list].abs().min(axis=1)
            df_final["winners"] = df_final[name_list].abs().idxmin(axis=1)
        # Value Counts of the dataframe
        if cell:
            print(
                "### Here are the results of the methods based on cell specific & benchmarked on {} ###\n\n".format(
                    statistic
                )
            )
        print(df_final["winners"].value_counts())
        display(df_final.sort_values("winners value"))

        sns.lineplot(data=df_final['SVR'],label='SVR')
        sns.lineplot(data=df_final['Kassandra'],label='Kassandra')
        sns.lineplot(data=df_final['cellanneal'],label='Cellanneal')
        sns.scatterplot(data=df_final['SVR'], legend=False)
        sns.scatterplot(data=df_final['Kassandra'], legend=False)
        sns.scatterplot(data=df_final['cellanneal'], legend=False)
        plt.xlabel("Cell Types")
        plt.ylabel(statistic)
        plt.title("{} across different Methods at Cell type level".format(statistic))
        plt.xticks(rotation = 90)
        plt.legend()
        plt.show()


# %%
