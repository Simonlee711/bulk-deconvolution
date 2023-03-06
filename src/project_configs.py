'''
File that contains some plotting parameter variables and other project configure files
'''

__author__ = 'Simon Lee (slee@celsiustx.com)'

#################################################
# Plotting Configs
#################################################
style = 'ggplot'
font_family  = 'sans-serif' 
font_serif = 'Ubuntu' 
font_monospace = 'Ubuntu Mono' 
font_size = 14 
axes_label_size = 12 
axes_label_weight = 'bold' 
axes_title_size = 12 
xtick_label_size = 12 
ytick_label_size = 12 
legend_font_size = 12 
figure_title_size = 12 
image_cmap= 'jet' 
image_interpolation = 'none' 
figure_size = (12, 10) 
axes_grid=True
lines_line_width = 2 
lines_marker_size = 8

#################################################
# Cell specific colors
#################################################

cells_p = {'B_cells': '#558ce0',
                'CD4_T_cells': '#28a35c',
                'CD8_T_cells': '#58d3bb',
                'Dendritic_cells': '#eaabcc',
                'Endothelium': '#F6783E',
                'Fibroblasts': '#a3451a',
                'Macrophages': '#d689b1',
                'Monocytes': '#ad4f80',
                'NK_cells': '#61cc5b',
                'Neutrophils': '#FCB586',
                'T_cells': '#808000',
                'Other': '#999999',
                'Immune_general': '#4f80ad',
                'Monocytic_cells': '#61babf',
                'Lymphocytes': '#5e9e34',
                'Plasma_B_cells': '#29589e',
                'Non_plasma_B_cells': '#248ce0',
                'Granulocytes': '#FCB511',
                'Basophils': '#fa7005',
                'Eosinophils': '#aaB586',
                'Naive_CD4_T_cells': '#bf8f0f',
                'Memory_CD4_T_cells': '#0f8f0f',
                'Memory_B_cells': '#55afe0',
                'Naive_CD8_T_cells': '#8FBC8F',
                'Memory_CD8_T_cells': '#8FBCFF',
                'Naive_B_cells': '#558cff'}