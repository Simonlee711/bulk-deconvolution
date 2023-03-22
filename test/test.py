'''
Test script to test out benchmark
'''
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from deconv import Deconvolution

x = Deconvolution()
pred1 = pd.read_csv('pred.csv', index_col=0)
pred2 = pd.read_csv('pred2.csv',index_col=0)
pred3 = pd.read_csv('pred3.csv',index_col=0)

pred3 = pred2-0.6
pred3.to_csv("pred3.csv")

# oops
ca1 = pred1.T.copy()
ca1.loc['B_cells'] = ca1.loc[['B', 'B-naive']].sum()
ca1.loc['CD4_T_cells'] = ca1.loc[['CD4', 'CD4-naive']].sum()
ca1.loc['CD8_T_cells'] = ca1.loc[['CD8']].sum()
ca1.loc['NK_cells'] = ca1.loc[['NK']].sum()
ca1.loc['Tregs'] = ca1.loc[['Treg']].sum()
ca1.loc['T_cells'] = ca1.loc[['CD8_T_cells', 'CD4_T_cells', 'Tregs', 'T_undef']].sum()
ca1.loc['Lymphocytes'] = ca1.loc[['B_cells', 'T_cells', 'NK_cells']].sum()

# 2
ka1 = pred2.T.copy()
ka1.loc['B_cells'] = ka1.loc[['B', 'B-naive']].sum()
ka1.loc['CD4_T_cells'] = ka1.loc[['CD4', 'CD4-naive']].sum()
ka1.loc['CD8_T_cells'] = ka1.loc[['CD8']].sum()
ka1.loc['NK_cells'] = ka1.loc[['NK']].sum()
ka1.loc['Tregs'] = ka1.loc[['Treg']].sum()
ka1.loc['T_cells'] = ka1.loc[['CD8_T_cells', 'CD4_T_cells', 'Tregs', 'T_undef']].sum()
ka1.loc['Lymphocytes'] = ka1.loc[['B_cells', 'T_cells', 'NK_cells']].sum()
#3
svr1 = pred3.T.copy()
svr1.loc['B_cells'] = svr1.loc[['B', 'B-naive']].sum()
svr1.loc['CD4_T_cells'] = svr1.loc[['CD4', 'CD4-naive']].sum()
svr1.loc['CD8_T_cells'] = svr1.loc[['CD8']].sum()
svr1.loc['NK_cells'] = svr1.loc[['NK']].sum()
svr1.loc['Tregs'] = svr1.loc[['Treg']].sum()
svr1.loc['T_cells'] = svr1.loc[['CD8_T_cells', 'CD4_T_cells', 'Tregs', 'T_undef']].sum()
svr1.loc['Lymphocytes'] = svr1.loc[['B_cells', 'T_cells', 'NK_cells']].sum()


preds = [ca1, ka1, svr1]
true = pd.read_csv('../data/GSE107572_cytof.tsv.tar.gz', sep='\t', index_col=0)
x.benchmark(preds, true, statistic='r2')
# %%
