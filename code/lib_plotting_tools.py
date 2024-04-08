#! /usr/bin/env python3

import seaborn as sns 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

from typing import Callable, Union

from dataclasses import dataclass

def add_colorbar(ax, cmap, norm: Callable, label: str):
    #Here we add a color bar for the interpretation of the barplot colors
    sm = plt.cm.ScalarMappable(cmap=cmap)#, norm=norm)
    sm.set_array([])
    colorbar = plt.colorbar(mappable=sm, ax=ax)

    #tick_names = name_classes if color_class==1 else name_classes[::-1]
    #colorbar.ax.set_yticklabels(tick_names)
    #colorbar.set_ticks([0,1])
    
    colorbar.ax.tick_params(color='#FFFFFF', pad=0)
    colorbar.set_label(label, rotation=90, ha='center',va='center')
    return colorbar


@dataclass
class ColorMapWithNorm():
    cmap: Union[mcolors.LinearSegmentedColormap, mcolors.ListedColormap, sns.palettes._ColorPalette]
    norm: Callable


def get_divergent_colormap(min_value=-3, max_value=3, center=0, cmap_name='coolwarm') -> ColorMapWithNorm:
    most_distant_absval = max(center-min_value, max_value-center)
    norm_func = mpl.colors.Normalize(center - most_distant_absval, center + most_distant_absval)
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    return ColorMapWithNorm(cmap=cmap, norm=norm_func)


def get_color_from_value(value: float, cmap, norm_func: Callable) -> str:
    normalized_value = norm_func(value)
    color = cmap(normalized_value)
    hex_color = mcolors.to_hex(color)
    return hex_color



