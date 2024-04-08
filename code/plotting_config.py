import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

import numpy as np


mpl.rcParams.update({'figure.autolayout':True})

sns.set(style="whitegrid", rc={"grid.linestyle": "--"})

husl = sns.color_palette('husl',32)
sns.palplot(husl)
plt.close()
#plt.style.use('seaborn-notebook')

SMALL_SIZE=13
MEDIUM_SIZE=15
BIGGER_SIZE=17

mpl.rc('font',size=SMALL_SIZE)
mpl.rc('axes',titlesize=SMALL_SIZE)
mpl.rc('axes',labelsize=MEDIUM_SIZE)
mpl.rc('xtick',labelsize=SMALL_SIZE)
mpl.rc('ytick',labelsize=SMALL_SIZE)
mpl.rc('legend',fontsize=SMALL_SIZE)
mpl.rc('figure',titlesize=BIGGER_SIZE)

def plot_resizelabel(label):
    """ If label is a string with multiple spaces and more than 20 char, add '\n' in middle.
    """
    if len(label)>20 and label.count(' ')>=3:
        tmp = label.split(' ')
        mid = int(np.ceil(len(tmp)/2))
        new_label = ' '.join(tmp[:mid])+'\n'+' '.join(tmp[mid:])
    else:
        new_label=label
    return new_label



def create_regular_grid_axes(N_tot, N_cols, height_row, width):
    """ Create a figure with a grid of N_tot axes with N_cols.
    
    Given N_tot a number of axes to generate, and N_cols the maximum
    number of axes on a row, this function generates the list of axes
    with the correct layout.
    
    To define the size of the figure, height_row and width are used.
    
    In:
        N_tot (int):
        N_cols (int):
        height_row (int):
        width (int):
        
    Return:
        (fig, axs)
        
    """
    N_rows = N_tot // N_cols
    N_rows += N_tot % N_cols
    Position = range(1, N_tot+1)
    
    fig = plt.figure(figsize = (width, height_row * N_rows))
    axs = []
    for pos in Position:
        axs.append(fig.add_subplot(N_rows, N_cols, pos))
        
    return (fig, axs)







def my_savefig(savefig_file, ext_list=['svg','png','pdf'], bbox_inches='tight'):
    for ext in ext_list:
        plt.savefig(savefig_file.format(EXT=ext))

