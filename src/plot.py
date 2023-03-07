'''
A module that includes some benchmarking plotting functions

Classes:
    Plot
'''

__author__ = 'Simon Lee (slee@celsiustx.com)'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import project_configs as project_configs
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from stats import statsTest

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

class Plot():
    '''
    class that performs plotting features for better visualization of our results

    Functions:

        stack_plot(df)
        heat_map(mix_df)
        corr_plot(predicted_values, true_values) -> axs 
        print_cell(predicted_values, true_values) -> ax
        print_fitted_line(predicted_values, true_values, ...) -> ax
        print_cell_whole(predicted_values, true_values, ...) -> ax
        bland_altman(predicted_values, true_values, ...) -> ax
        bland_altman2(predicted_values, true_values, ...) -> ax
        get_cmap(n, name='hsv') -> ax.pallete
        plot_sample(predicted_values, true_values, ...) -> ax
        print_sample(predicted_values, true_values, ...) -> ax
        balnd_altman_v2(predicted_values, true_values, ...) -> ax
        bland_altmanv2_2(predicted_values, true_values, ...) -> ax
        qq(predicted_values, true_values) - NOT IMPLEMENTED YET
    '''

    def stack_plot(self,df):
        '''
        stack plots to visualize the cellular composition

        Parameters:
            df (pandas.DataFrame): dataframes of cellular proportions
        '''
        corr_list = []
        if 'rho_Spearman' in df.columns:
            corr_list.append('rho_Spearman')
        if 'rho_Pearson' in df.columns:
            corr_list.append('rho_Pearson')
        if len(corr_list) > 0:
            df = df.drop(corr_list, axis=1)
        else:
            df = df
        df.plot(kind='bar', stacked=True)
        plt.title("Cell Deconvolutions")
        plt.xlabel("Bulk Sample ID")
        plt.ylabel("Cell type Proportions (%)")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def heat_map(self,mix_df):
        '''
        Heat Map representation of the cell proportion. Very elegant compared to stack plots
    
        Parameters:
            mix_df (pandas.DataFrame): dataframes of cellular proportions
        '''
        corr_list = []
        if 'rho_Spearman' in mix_df.columns:
            corr_list.append('rho_Spearman')
        if 'rho_Pearson' in mix_df.columns:
            corr_list.append('rho_Pearson')
        if len(corr_list) > 0:
            plot_df = mix_df.drop(corr_list, axis=1)
        else:
            plot_df = mix_df
        fig, ax = plt.subplots(
        figsize=(10/np.shape(mix_df)[1]*np.shape(mix_df)[0],
                    0.5*np.shape(mix_df)[1]))
        # removing correlation coefficients from df
        corr_list = []
        if 'rho_Spearman' in mix_df.columns:
            corr_list.append('rho_Spearman')
        if 'rho_Pearson' in mix_df.columns:
            corr_list.append('rho_Pearson')
        if len(corr_list) > 0:
            plot_df = mix_df.drop(corr_list, axis=1)
        else:
            plot_df = mix_df
        
        # plott

        ax = sns.heatmap(plot_df.T.sort_index(0),
                        linewidths=.5, square=True, cmap='viridis',
                        cbar_kws={'shrink': 0.7, 'label': 'fraction'})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        bottom, top = ax.get_ylim()
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.invert_yaxis()
        plt.title("Cell Deconvolutions")
        plt.xlabel("Bulk Sample ID")
        plt.ylabel("Cell types")
        #plt.savefig('', bbox_inches='tight')

    def plot_cell(self, predicted_values, true_values, figsize=(24, 24), pallete=None):
        '''
        Author: Kassandra
        
        The function draws a grid of scatterplots for each cell type with colors from pallete,
        with correlation in titles and straight lines based on linear regression if needed.

        Parameters:
            predicted_values (pandas.DataFrame): dataframes of predicted cellular proportions
            true_values (pandas.DataFrame): dataframes of "True" Flow Cytometry or Pathology quanitfied matrices
            figsize (set): a set that specifies figure size. Default (24, 24) 
            pallete (list): a color pallete for our plotting. Default is set to None 
        
        Returns:
            axs subplot figure
        '''
        ind_names = predicted_values.index.intersection(true_values.index)
        col_names = predicted_values.columns.intersection(true_values.columns)
        predicted_values = predicted_values.loc[ind_names, col_names]
        true_values = true_values.loc[ind_names, col_names]

        if len(ind_names) < 4:
            num_ncols = len(ind_names)
        else:
            num_ncols = 4

        num_nrows = (len(ind_names) - 1) // 6 + 1
        
        figsize = (figsize[0], figsize[0] * num_nrows / num_ncols)

        fig, axs = plt.subplots(num_nrows, num_ncols, figsize=figsize)
        #fig.set_title("Correlation of predicted and real percentage of cells", fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.91, wspace=0.4, hspace=0.4)

        
        ordered_names = ind_names.sort_values()

        for ax, cell in zip(axs.flat, ordered_names):
            
            sub_title = cell
            if cell in pallete:
                colors = pallete[cell]
            else:
                colors = '#999999'
            self.print_cell(predicted_values.loc[cell], true_values.loc[cell], ax=ax,title=sub_title, pallete=pallete, color = colors)

        return axs

    def print_cell(self, predicted_values, true_values, ax=None, pallete=None, single_color='#999999',
                predicted_name='Predicted percentage of cells, %',
                true_name='Real percentage of cells, %', title=None, corr_title=True, corr_rounding=3,
                figsize=(6, 6), s=60, title_font=20, labels_font=18, ticks_size=17, xlim=None, ylim=None,
                corr_line=True, linewidth=1, line_color='black', pad=15, min_xlim=0.1, min_ylim=0.1, labelpad=None, color = None):
        """
        Author: Kassandra

        The function draws a scatterplot with colors from pallete, with correlation in title and
        straight line based on linear regression on x_values and y_values if needed.

        Parameters:
            predicted_values: pandas Series
            true_values: pandas Series
            ax: matplotlib.axes
            pallete: dict with colors, keys - some or all names from predicted_values and true_values index
            single_color: what color to use if there is no palette or some names are missed
            predicted_name: xlabel for plot
            true_name: ylabel for plot
            title: title for plot, will be combined with ', corr = ' if corr_title=True
            corr_title: whether to calculate Pearson correlation and print it with title
            corr_rounding: precision in decimal digits for correlation if corr_title=True
            figsize: figsize if ax=None
            s: scalar or array_like, shape (n, ), marker size for scatter
            title_font: title size
            labels_font: xlabel and ylabel size
            ticks_size: tick values size
            xlim: x limits, if None xlim will be (0, 1.2 * max(predicted_values))
            ylim: y limits, if None ylim will be (0, 1.2 * max(true_values))
            corr_line: whether to draw a straight line based on linear regression on x_values and y_values
            linewidth: width of the fitted line
            line_color: color of the fitted line
            pad: distance for titles from picture
            min_xlim: minimal range (max picture value) for x
            min_ylim: minimal range (max picture value) for y

        Returns:
            axs subplot figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ind_points = true_values.dropna().index.intersection(predicted_values.dropna().index)
        ax.grid(b=False)
        predicted_values = predicted_values.loc[ind_points].astype(float)
        true_values = true_values.loc[ind_points].astype(float)
        if corr_title:
            corrcoef, pval = pearsonr(predicted_values, true_values)
            corrcoef = str(round(corrcoef, corr_rounding))
            pval = str(round(pval, 3))
            if title is not None:
                ax.set_title('{title}, corr = {corr}\np = {p}'.format(title=title,
                                                                    corr=corrcoef,
                                                                    p=pval),
                            size=title_font, pad=pad)
            else:
                ax.set_title('Corr = {corr}\np = {p}'.format(corr=corrcoef,
                                                            p=pval),
                            size=title_font, pad=pad)
        elif title is not None:
            ax.set_title(title, size=title_font, pad=pad)
        ax.set_xlabel(predicted_name, size=labels_font, labelpad = labelpad)
        ax.set_ylabel(true_name, size=labels_font, labelpad = labelpad)
        ax.tick_params(labelsize=ticks_size)

        ax.set_xlim(-0.5, max(1.2 * max(predicted_values), min_xlim))
        ax.set_ylim(-0.5, max(1.2 * max(true_values), min_ylim))

        if single_color != '#999999':
            ax.scatter(predicted_values, true_values, s=s, c=single_color)
        else:
            ax.scatter(predicted_values, true_values, s=s, c=color)

        if corr_line:
            self.print_fitted_line(predicted_values, true_values, ax=ax, linewidth=linewidth, line_color=line_color)

        return ax
    
    def print_fitted_line(self, x_values, y_values, ax=None, linewidth=1, line_color='black', figsize=(6, 6)):
        """
        Author: Kassandra

        The function draws a straight line based on linear regression on x_values and y_values.

        Parameters:
            x_values: pandas Series or numpy array
            y_values: pandas Series or numpy array
            ax: matplotlib.axes
            linewidth: width of the line
            line_color: color of the line
            figsize: figsize if ax=None
        
        Returns:
            Ax subplot figure with linear regression line fitted
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        fit_coefs = np.polyfit(x_values, y_values, deg=1)
        fit_values = np.sort(x_values)
        ax.plot(fit_values, fit_coefs[0] * fit_values + fit_coefs[1], linewidth=linewidth, color=line_color)

        return ax
    
    def plot_whole(self, predicted_values, true_values, ax=None, pallete=None, single_color='#999999',
                           colors_by='index', predicted_name='Predicted percentage of cells, %',
                           true_name='Real percentage of cells, %', 
                           title=None, stat=True,
                           corr_rounding=3, figsize=(8, 8), s=50, title_font=20,
                           labels_font=20, ticks_size=17, xlim=None, ylim=None,
                           corr_line=True, linewidth=1, line_color='black', pad=15, min_xlim=10, min_ylim=10):
        '''
        Author: Kassandra

        The function draws a scatterplot for all cell types with colors from pallete, with correlation in title and
        straight line based on linear regression if needed.

        Parameters:
            predicted_values: pandas Series
            true_values: pandas Series
            ax: matplotlib.axes
            pallete: dict with colors, keys - some or all names from predicted_values and true_values index
            single_color: what color to use if there is no palette or some names are missed
            predicted_name: xlabel for plot
            true_name: ylabel for plot
            title: title for plot, will be combined with ', corr = ' if corr_title=True
            corr_title: whether to calculate Pearson correlation and print it with title
            corr_rounding: precision in decimal digits for correlation if corr_title=True
            figsize: figsize if ax=None
            s: scalar or array_like, shape (n, ), marker size for scatter
            title_font: title size
            labels_font: xlabel and ylabel size
            ticks_size: tick values size
            xlim: x limits, if None xlim will be (0, 1.2 * max(predicted_values))
            ylim: y limits, if None ylim will be (0, 1.2 * max(true_values))
            corr_line: whether to draw a straight line based on linear regression on x_values and y_values
            linewidth: width of the fitted line
            line_color: color of the fitted line
            pad: distance for titles from picture
            min_xlim: minimal range (max picture value) for x
            min_ylim: minimal range (max picture value) for y
        
        Returns:
            ax Figure

        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ind_names = predicted_values.index.intersection(true_values.index)
        col_names = predicted_values.columns.intersection(true_values.columns)
        predicted_values = predicted_values.loc[ind_names, col_names]
        true_values = true_values.loc[ind_names, col_names]
        ravel_predicted = pd.Series(predicted_values.values.ravel()).dropna()
        ravel_true = pd.Series(true_values.values.ravel()).dropna()
        ravel_ind = ravel_predicted.index.intersection(ravel_true.index)
        ravel_predicted = ravel_predicted.loc[ravel_ind].astype(float)
        ravel_true = ravel_true.loc[ravel_ind].astype(float)
        if xlim is None:
            xlim = (0, max(1.2 * max(ravel_predicted), min_xlim))
        if ylim is None:
            ylim = (0, max(1.2 * max(ravel_true), min_ylim))
        
        if not title:
            title = ''
        
        if stat:
            stats = statsTest()
            rmse, corr, p, r2 = stats.test(ravel_predicted.loc[ravel_ind], ravel_true.loc[ravel_ind])
        
        ax.set_title(title, size=title_font, pad=pad)

        for col in col_names:
            if colors_by == 'index':
                ax_pallete = pallete
                ax_single_color = single_color
            elif col in pallete:
                ax_pallete = None
                ax_single_color = pallete[col]
            else:
                ax_pallete = None
                ax_single_color = single_color

            self.print_cell(predicted_values[col], true_values[col], ax=ax, pallete=ax_pallete, single_color=ax_single_color,
                    predicted_name=predicted_name, true_name=true_name, corr_title=False, s=s, labels_font=labels_font,
                    ticks_size=ticks_size, xlim=xlim, ylim=ylim, corr_line=False)
        if corr_line:
            self.print_fitted_line(ravel_predicted, ravel_true, ax=ax, linewidth=linewidth, line_color=line_color)

        if stat:  
            bbox = dict(boxstyle ="round", fc ="0.8")
            ax.annotate('Statistical Tests:\nRMSE: %.2f\nPears Corr: %.2f\nP value: %.2f\nR^2: %.2f' % (rmse,corr,p,r2), xy=(1, 0), xycoords='axes fraction', fontsize=16,
                xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom', bbox = bbox)

        return ax

    def bland_altman(self, predicted_values, true_values, figsize=(24, 24), pallete=None):
        '''
        Author: Simon Lee

        plots residuals and shows biases among different cell groups. Shows off which sample in the cell group under predict, over predict and shows off mean + std deviations of each cell type

        Parameters:
            predicted_values (pandas.DataFrame): dataframes of predicted cellular proportions
            true_values (pandas.DataFrame): dataframes of "True" Flow Cytometry or Pathology quanitfied matrices
            figsize (set): a set that specifies figure size. Default (24, 24) 
            pallete (list): a color pallete for our plotting. Default is set to None 
        
        Returns:
            axs subplot figure        
        '''
        
        ind_names = predicted_values.index.intersection(true_values.index)
        col_names = predicted_values.columns.intersection(true_values.columns)
        predicted_values = predicted_values.loc[ind_names, col_names]
        true_values = true_values.loc[ind_names, col_names]

        if len(ind_names) < 4:
            num_ncols = len(ind_names)
        else:
            num_ncols = 4

        num_nrows = (len(ind_names) - 1) // 6 + 1
        
        figsize = (figsize[0], figsize[0] * num_nrows / num_ncols)
        fig, axs = plt.subplots(num_nrows, num_ncols, figsize=figsize)
        #fig.set_title("Correlation of predicted and real percentage of cells", fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.91, wspace=0.4, hspace=0.4)

        
        ordered_names = ind_names.sort_values()
        for ax, cell in zip(axs.flat, ordered_names):
            
            sub_title = cell
            if cell in pallete:
                colors = pallete[cell]
            else:
                colors = '#999999'
            
            self.bland_altman2(predicted_values.loc[cell], true_values.loc[cell], ax=ax,title=sub_title, color = colors)

        return axs

    def bland_altman2(self, predicted_values, true_values, ax=None, pallete=project_configs.cells_p, single_color='#999999',
                predicted_name='Average (%), (pred+true)/2',
                true_name='Difference (%) (pred-true)', title=None, corr_title=True, corr_rounding=3,
                figsize=(6, 6), s=60, title_font=15, labels_font=12, ticks_size=17, xlim=None, ylim=None,
                corr_line=True, linewidth=1, line_color='black', pad=15, min_xlim=0.1, min_ylim=0.1, labelpad=None, color = None):
        '''
        Author: Simon Lee

        bland altman helper function. Where it calculate average, plots residuals and annotates the graphs with averages & std. deviations

        Parameters:
            predicted_values: pandas Series
            true_values: pandas Series
            ax: matplotlib.axes
            pallete: dict with colors, keys - some or all names from predicted_values and true_values index
            single_color: what color to use if there is no palette or some names are missed
            predicted_name: xlabel for plot
            true_name: ylabel for plot
            title: title for plot, will be combined with ', corr = ' if corr_title=True
            corr_title: whether to calculate Pearson correlation and print it with title
            corr_rounding: precision in decimal digits for correlation if corr_title=True
            figsize: figsize if ax=None
            s: scalar or array_like, shape (n, ), marker size for scatter
            title_font: title size
            labels_font: xlabel and ylabel size
            ticks_size: tick values size
            xlim: x limits, if None xlim will be (0, 1.2 * max(predicted_values))
            ylim: y limits, if None ylim will be (0, 1.2 * max(true_values))
            corr_line: whether to draw a straight line based on linear regression on x_values and y_values
            linewidth: width of the fitted line
            line_color: color of the fitted line
            pad: distance for titles from picture
            min_xlim: minimal range (max picture value) for x
            min_ylim: minimal range (max picture value) for y

        Returns:
            ax subplot figure
        '''


        ind_points = true_values.dropna().index.intersection(predicted_values.dropna().index)
        ax.grid(b=False)
        predicted_values = predicted_values.loc[ind_points].astype(float)
        true_values = true_values.loc[ind_points].astype(float)

        # so the percentages line up

        df = pd.merge(predicted_values, true_values, left_index=True, right_index=True)
        indices = df.columns
        df['diff'] = df[indices[0]] - df[indices[1]]
        df['avg'] = (df[indices[0]] + df[indices[1]])/2

        title = 'Bland Altman of ' + title + ': {' + str(round(df['diff'].mean(),3)) + ' +/- ' + str(round(df['diff'].std(),3)) + '}'

        ax.set_title(title, size=title_font, pad=pad)
        ax.set_xlabel(predicted_name, size=labels_font, labelpad = labelpad)
        ax.set_ylabel(true_name, size=labels_font, labelpad = labelpad)
        ax.tick_params(labelsize=ticks_size)

        maximum = max(1.2 * max(predicted_values), min_xlim)

        ax.set_xlim(-0.5,maximum*1.5)
        ax.set_ylim(df['diff'].mean() -4* df['diff'].std(),df['diff'].mean() + 3* df['diff'].std())
        
        ax.scatter(df['avg'], df['diff'], s=s, c=color)

        # plot mean lines and std deviations
        ax.axhline(df['diff'].mean(), c='#000000')
        ax.axhline(df['diff'].mean() + 2* df['diff'].std(), ls='--', c='#000000')
        ax.axhline(df['diff'].mean() + df['diff'].std(), ls='dashdot', c='#000000')
        ax.axhline(df['diff'].mean() - df['diff'].std(), ls='dashdot', c='#000000')
        ax.axhline(df['diff'].mean() - 2* df['diff'].std(), ls='--', c='#000000')

        ax.annotate('-SD2: %.2f' % (df['diff'].mean() - 2* df['diff'].std()), xy=(1, 0), xycoords='axes fraction', fontsize=16,
                xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom')
        
        ax.annotate('+SD2: %.2f' % (df['diff'].mean() + 2* df['diff'].std()), xy=(0.97, 0.9), xycoords='axes fraction', fontsize=16,
                xytext=(5, 5), textcoords='offset points', ha='right', va='top')

        del df
        
        return ax

    def get_cmap(self, n, name='hsv'):
        '''
        Author Stack Overflow

        Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.

        Parameters:
            n (int): number of colors that need to be generated
            name (string): data type for color pallete
        
        Returns:
            color pallete
        '''
        return plt.cm.get_cmap(name, n)
    
    def plot_sample(self, predicted_values, true_values, figsize=(24, 24), pallete=None, specific_col = None):
        '''
        Author: modified version of Kassandra plot that plots at a sample level instead of a cell level

        Parameters:
            predicted_values (pandas.DataFrame): dataframes of predicted cellular proportions
            true_values (pandas.DataFrame): dataframes of "True" Flow Cytometry or Pathology quanitfied matrices
            figsize (set): a set that specifies figure size. Default (24, 24) 
            pallete (list): a color pallete for our plotting. Default is set to None 
        
        Returns:
            axs subplot figure        
        '''
        ind_names = predicted_values.index.intersection(true_values.index)
        if specific_col != None:
            ind_points = true_values[specific_col].dropna().index.intersection(predicted_values[specific_col].dropna().index)
        else:
            ind_points = true_values.dropna().index.intersection(predicted_values.dropna().index)
        col_names = predicted_values.columns.intersection(true_values.columns)
        predicted_values = predicted_values.loc[ind_names, col_names]
        true_values = true_values.loc[ind_names, col_names]

        if len(col_names) < 4:
            num_ncols = len(col_names)
        else:
            num_ncols = 4

        num_nrows = (len(col_names) - 1) // 6 + 1
        
        figsize = (figsize[0], figsize[0] * num_nrows / num_ncols)

        fig, axs = plt.subplots(num_nrows, num_ncols, figsize=figsize)
        #fig.set_title("Correlation of predicted and real percentage of cells", fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.91, wspace=0.4, hspace=0.4)

        predicted_values = predicted_values.T
        true_values = true_values.T

        
        ordered_names = col_names.sort_values()
        ordered_cell_names = ind_names.sort_values()

        cmap = pallete
        counter = 0
        
        for ax, sample in zip(axs.flat, ordered_names):
            
            sub_title = sample

            if counter == len(ordered_names)-1:
                self.print_sample(predicted_values.loc[sample], true_values.loc[sample], ax=ax,title=sub_title, color = cmap, cells = ordered_cell_names, legend = True)
            else:
                self.print_sample(predicted_values.loc[sample], true_values.loc[sample], ax=ax,title=sub_title, color = cmap, cells = ordered_cell_names)
            counter += 1
        
        # making a legend for plots
        if specific_col != None:
            for i, cell in enumerate(ind_points):
                plt.scatter([],[],color = cmap(i), label=cell)
        else: 
            for i, cell in enumerate(ind_points):
                plt.scatter([],[],color = cmap(i), label=cell)
        
        fig.legend(bbox_to_anchor=(1.2, 0.5),title="Cell types", prop={'size': 14})

        return axs

    def print_sample(self, predicted_values, true_values, ax=None, pallete=None, single_color='#999999',
                predicted_name='Predicted percentage of cells, %',
                true_name='Real percentage of cells, %', title=None, corr_title=True, corr_rounding=3,
                figsize=(6, 6), s=60, title_font=20, labels_font=18, ticks_size=17, xlim=None, ylim=None,
                corr_line=True, linewidth=1, line_color='black', pad=15, min_xlim=0.1, min_ylim=0.1, labelpad=None, color = None, cells = None, legend = False):
        """
        Author: Kassandra

        Modified version of print_cell() method that is now plotting at a sample level

        Parameters:
            predicted_values: pandas Series
            true_values: pandas Series
            ax: matplotlib.axes
            pallete: dict with colors, keys - some or all names from predicted_values and true_values index
            single_color: what color to use if there is no palette or some names are missed
            predicted_name: xlabel for plot
            true_name: ylabel for plot
            title: title for plot, will be combined with ', corr = ' if corr_title=True
            corr_title: whether to calculate Pearson correlation and print it with title
            corr_rounding: precision in decimal digits for correlation if corr_title=True
            figsize: figsize if ax=None
            s: scalar or array_like, shape (n, ), marker size for scatter
            title_font: title size
            labels_font: xlabel and ylabel size
            ticks_size: tick values size
            xlim: x limits, if None xlim will be (0, 1.2 * max(predicted_values))
            ylim: y limits, if None ylim will be (0, 1.2 * max(true_values))
            corr_line: whether to draw a straight line based on linear regression on x_values and y_values
            linewidth: width of the fitted line
            line_color: color of the fitted line
            pad: distance for titles from picture
            min_xlim: minimal range (max picture value) for x
            min_ylim: minimal range (max picture value) for y

        Returns:
            axs subplot figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ind_points = true_values.dropna().index.intersection(predicted_values.dropna().index)
        ax.grid(b=False)
        predicted_values = predicted_values.loc[ind_points].astype(float)
        true_values = true_values.loc[ind_points].astype(float)
        ordered_cell = ind_points.sort_values()
        if corr_title:
            corrcoef, pval = pearsonr(predicted_values, true_values)
            corrcoef = str(round(corrcoef, corr_rounding))
            pval = str(round(pval, 3))
            if title is not None:
                ax.set_title('{title}, corr = {corr}\np = {p}'.format(title=title,
                                                                    corr=corrcoef,
                                                                    p=pval),
                            size=title_font, pad=pad)
            else:
                ax.set_title('Corr = {corr}\np = {p}'.format(corr=corrcoef,
                                                            p=pval),
                            size=title_font, pad=pad)
        elif title is not None:
            ax.set_title(title, size=title_font, pad=pad)
        ax.set_xlabel(predicted_name, size=labels_font, labelpad = labelpad)
        ax.set_ylabel(true_name, size=labels_font, labelpad = labelpad)
        ax.tick_params(labelsize=ticks_size)

        ax.set_xlim(-0.5, max(1.2 * max(predicted_values), min_xlim))
        ax.set_ylim(-0.5, max(1.2 * max(true_values), min_ylim))

        if legend:
            for i, cell in enumerate(ordered_cell):
                ax.scatter(predicted_values[i], true_values[i], s=s, c=color(i), label = cells[i])
        else:
            for i, cell in enumerate(ordered_cell):
                ax.scatter(predicted_values[i], true_values[i], s=s, c=color(i))

        if corr_line:
            self.print_fitted_line(predicted_values, true_values, ax=ax, linewidth=linewidth, line_color=line_color)

        return ax

    def bland_altman_v2(self, predicted_values, true_values, figsize=(24, 24), pallete=None, specific_col = None):
        '''
        Author: Simon Lee

        plots residuals and shows biases among different sample groups. Shows off which cells under predict, over predict and shows off mean + std deviations of each cell type

        Parameters:
            predicted_values (pandas.DataFrame): dataframes of predicted cellular proportions
            true_values (pandas.DataFrame): dataframes of "True" Flow Cytometry or Pathology quanitfied matrices
            figsize (set): a set that specifies figure size. Default (24, 24) 
            pallete (list): a color pallete for our plotting. Default is set to None 
        
        Returns:
            axs subplot figure   
        
        '''
        
        ind_names = predicted_values.index.intersection(true_values.index)

        if specific_col != None:
            ind_points = true_values[specific_col].dropna().index.intersection(predicted_values[specific_col].dropna().index)
        else:
            ind_points = true_values.dropna().index.intersection(predicted_values.dropna().index)

        col_names = predicted_values.columns.intersection(true_values.columns)
        predicted_values = predicted_values.loc[ind_names, col_names]
        true_values = true_values.loc[ind_names, col_names]

        if len(col_names) < 4:
            num_ncols = len(col_names)
        else:
            num_ncols = 4

        num_nrows = (len(col_names) - 1) // 6 + 1
        
        figsize = (figsize[0], figsize[0] * num_nrows / num_ncols)
        fig, axs = plt.subplots(num_nrows, num_ncols, figsize=figsize)
        #fig.set_title("Correlation of predicted and real percentage of cells", fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.91, wspace=0.4, hspace=0.4)


        predicted_values = predicted_values.T
        true_values = true_values.T
        
        ordered_names = col_names.sort_values()
        ordered_cell_names = ind_names.sort_values()
        cmap = pallete

        counter = 0

        for ax, sample in zip(axs.flat, ordered_names):
            
            sub_title = sample

            if counter == len(ordered_names)-1:
                self.bland_altmanv2_2(predicted_values.loc[sample], true_values.loc[sample], ax=ax,title=sub_title,color = cmap, cells = ordered_cell_names, legend=True)
            else:
                self.bland_altmanv2_2(predicted_values.loc[sample], true_values.loc[sample], ax=ax,title=sub_title,color = cmap, cells = ordered_cell_names)
            counter += 1
        
        # making a legend for plots
        if specific_col != None:
            for i, cell in enumerate(ind_points):
                plt.scatter([],[],color = cmap(i), label=cell)
        else: 
            for i, cell in enumerate(ind_points):
                plt.scatter([],[],color = cmap(i), label=cell)
        
        fig.legend(bbox_to_anchor=(1.2, 0.5),title="Cell types", prop={'size': 14})

        return axs

    def bland_altmanv2_2(self, predicted_values, true_values, ax=None, pallete=project_configs.cells_p, single_color='#999999',
                predicted_name='Average (%), (pred+true)/2',
                true_name='Difference (%) (pred-true)', title=None, corr_title=True, corr_rounding=3,
                figsize=(6, 6), s=60, title_font=15, labels_font=12, ticks_size=17, xlim=None, ylim=None,
                corr_line=True, linewidth=1, line_color='black', pad=15, min_xlim=0.1, min_ylim=0.1, labelpad=None, color = None, cells = None, legend=False):
        
        '''       
        Author: Simon Lee

        bland altman helper function. Where it calculate average, plots residuals and annotates the graphs with averages & std. deviations

        Parameters:
            predicted_values: pandas Series
            true_values: pandas Series
            ax: matplotlib.axes
            pallete: dict with colors, keys - some or all names from predicted_values and true_values index
            single_color: what color to use if there is no palette or some names are missed
            predicted_name: xlabel for plot
            true_name: ylabel for plot
            title: title for plot, will be combined with ', corr = ' if corr_title=True
            corr_title: whether to calculate Pearson correlation and print it with title
            corr_rounding: precision in decimal digits for correlation if corr_title=True
            figsize: figsize if ax=None
            s: scalar or array_like, shape (n, ), marker size for scatter
            title_font: title size
            labels_font: xlabel and ylabel size
            ticks_size: tick values size
            xlim: x limits, if None xlim will be (0, 1.2 * max(predicted_values))
            ylim: y limits, if None ylim will be (0, 1.2 * max(true_values))
            corr_line: whether to draw a straight line based on linear regression on x_values and y_values
            linewidth: width of the fitted line
            line_color: color of the fitted line
            pad: distance for titles from picture
            min_xlim: minimal range (max picture value) for x
            min_ylim: minimal range (max picture value) for y
            legend: Adds a legend which indicates cell types for each plot

        Returns:
            ax subplot figure
        '''

        ind_points = true_values.dropna().index.intersection(predicted_values.dropna().index)
        ax.grid(b=False)
        predicted_values = predicted_values.loc[ind_points].astype(float)
        true_values = true_values.loc[ind_points].astype(float)
        ordered_cell = ind_points.sort_values()

        # so the percentages line up

        df = pd.merge(predicted_values, true_values, left_index=True, right_index=True)
        indices = df.columns
        df['diff'] = df[indices[0]] - df[indices[1]]
        df['avg'] = (df[indices[0]] + df[indices[1]])/2

        title = 'Bland Altman of ' + title + ': {' + str(round(df['diff'].mean(),3)) + ' +/- ' + str(round(df['diff'].std(),3)) + '}'

        ax.set_title(title, size=title_font, pad=pad)
        ax.set_xlabel(predicted_name, size=labels_font, labelpad = labelpad)
        ax.set_ylabel(true_name, size=labels_font, labelpad = labelpad)
        ax.tick_params(labelsize=ticks_size)

        maximum = max(1.2 * max(predicted_values), min_xlim)

        ax.set_xlim(-0.5,maximum*1.5)
        ax.set_ylim(df['diff'].mean() -4* df['diff'].std(),df['diff'].mean() + 3* df['diff'].std())
        

        if legend:
            for i, cell in enumerate(ordered_cell):
                ax.scatter(df['avg'][i], df['diff'][i], s=s, c=color(i), label = cells[i])
        else:
            for i, cell in enumerate(ordered_cell):
                ax.scatter(df['avg'][i], df['diff'][i], s=s, c=color(i))
        

        # plot mean lines and std deviations
        ax.axhline(df['diff'].mean(), c='#000000')
        ax.axhline(df['diff'].mean() + 2* df['diff'].std(), ls='--', c='#000000')
        ax.axhline(df['diff'].mean() + df['diff'].std(), ls='dashdot', c='#000000')
        ax.axhline(df['diff'].mean() - df['diff'].std(), ls='dashdot', c='#000000')
        ax.axhline(df['diff'].mean() - 2* df['diff'].std(), ls='--', c='#000000')

        ax.annotate('-SD2: %.2f' % (df['diff'].mean() - 2* df['diff'].std()), xy=(1, 0), xycoords='axes fraction', fontsize=16,
                xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom')
        
        ax.annotate('+SD2: %.2f' % (df['diff'].mean() + 2* df['diff'].std()), xy=(0.97, 0.9), xycoords='axes fraction', fontsize=16,
                xytext=(5, 5), textcoords='offset points', ha='right', va='top')

        del df
        
        return ax

    def qq(self, predicted_values, true_values):
        '''
        Author Simon Lee

        Quantile Quantile plot. If the predicted and true arrays have similar probability distributions we should get a linear line but if there is some transformation along the x or y axis we know there is some technical bias involved indicating a "poor fit"
        '''
        pass    
    