'''
A module that calculates statistics for our cell deconvolution benchmarking

Classes:
    statsTest
'''

__author__ = 'Simon Lee (slee@celsiustx.com)'

import numpy as np
import project_configs as project_configs

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


class statsTest():
    '''
    class that performs all statistical test for the deconvolution methods

    Functions:

        rmse(predicted_values, true_values) -> float
        correlation(predicted_values, true_values, p = False) -> float
        Pvalue(predicted_values, true_values) -> float
        R_squared(predicted_values, true_values) -> float
        test(predicted_values, true_values) -> float, float, float, float
        '''
    
    def rmse(self, predicted_values, true_values):
        '''
        A method that calculates the Root mean squared error. 

         Parameters:
            predicted_values (nd.array): numpy array with predicted values
            true_values (nd.array): numpy array with true values

        Returns:
            rms (float): RMSE of two arrays
        '''
        rms = mse(true_values, predicted_values, squared=False)
        return rms
        
    def correlation(self,predicted_values, true_values, p = False):
        '''
        A method that calculates the pearson correlation coefficient

        Parameters:
            predicted_values (nd.array): numpy array with predicted values
            true_values (nd.array): numpy array with true values
            p (boolean): a flag to indicate whether to return p value or correlation coefficent. Automatically set at False

        Returns:
            corrcoef (float): Pearson Correlation Coeffiecient of two arrays
        '''
        corrcoef, pval = pearsonr(predicted_values, true_values)
        corrcoef = str(round(corrcoef, 3))

        if p:
            pval = str(round(pval, 3))
            return pval
        
        return corrcoef

    def Pvalue(self,predicted_values, true_values):
        '''
        A method that calculates the P value

        Parameters:
            predicted_values (nd.array): numpy array with predicted values
            true_values (nd.array): numpy array with true values

        Returns:
            pval (float): P-value of two arrays
        '''
        pval = self.correlation(predicted_values,true_values,p=True)
        return pval
        
    def R_squared(self, predicted_values, true_values):
        '''
        A method that calculates the R squared value

        Parameters:
            predicted_values (nd.array): numpy array with predicted values
            true_values (nd.array): numpy array with true values

        Returns:
            r2 (float): R-squared of two arrays
        '''
        r2 = r2_score(true_values, predicted_values)
        return r2
    
    def test(self, predicted_values, true_values):
        '''
        Runs all statistical tests in one method

        Parameters:
            predicted_values (nd.array): numpy array with predicted values
            true_values (nd.array): numpy array with true values

        Returns:
            Rmse (float): RMSE of two arrays
            Corr (float): Pearson Correlation (r) of two arrays
            Pval (float): P-value of two arrays
            R2 (float): R^2 of two arrays
        '''
        Rmse = self.rmse(predicted_values, true_values)
        Corr = self.correlation(predicted_values, true_values)
        Pval = self.Pvalue(predicted_values, true_values)
        R2 = self.R_squared(predicted_values, true_values)
        print("Statistical Tests\n------------------------\nRMSE:",round(Rmse,3), "\nPearson Correlation Coefficient:", round(float(Corr),3), "\nP value:",float(Pval),"\nR^2:", round(R2,3))
        return round(Rmse,3), round(float(Corr),3), round(float(Pval),3), round(R2,3)