import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import importlib

warnings.filterwarnings("ignore", category=UserWarning)

class HandlerRect(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        rect = mpatches.Rectangle([xdescent, ydescent], width, height, hatch='//', edgecolor='black', facecolor='white')
        return [rect]


def bar_chart(df1, df2, title):  
    color_palette = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf"   # cyan
    ]
    fig, ax = plt.subplots(figsize=(12, 3))
    bar_width = 0.2
    datasets = ["D" + str(i) for i in range(1, 21)]
    index = range(len(datasets))
    bars = []
    for i, (column, color) in enumerate(zip(df2.columns, color_palette)):
        bars.append(ax.bar([p + bar_width*i for p in index], df2[column], bar_width, label=column, color=color))
    ax.set_ylabel('')
    ax.set_title(title, fontsize=14)
    ax.set_xticks([p + 1.5 * bar_width for p in index])
    ax.set_xticklabels(datasets, rotation=45, ha='center')  
    sel_indices = df1.idxmin(axis=1)
    ax.minorticks_on()
    ax.tick_params(axis='y', which='minor', length=4)
    ax.tick_params(axis='x', which='minor', bottom=False) 
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    for i, sel_index in enumerate(sel_indices):
        bar_index = list(df1.columns).index(sel_index)
        bars[bar_index][i].set_hatch('//') 
    plt.tight_layout()
    plt.show()


def values_heatmap(df1, df2, methods, df1name="metric1", df2name="metric2", mycmap=["Blues", "Greens"]):
    importlib.reload(mpl)
    importlib.reload(plt)
    importlib.reload(sns)
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    sns.set(font_scale=1.6)
    sns.heatmap(df1, ax=axs[0], cmap=mycmap[0], annot=True, fmt=".4f")
    axs[0].set_xticks(np.arange(1, len(methods)+1)-0.5, methods, fontsize=16)
    axs[0].set_yticks(np.arange(1, df1.shape[0])-0.5, ["D" + str(i) for i in range(1, df1.shape[0])], fontsize=16, rotation=0, )
    axs[0].set_title(df1name)
    sns.heatmap(df2, ax=axs[1], cmap=mycmap[1], annot=True, fmt=".4f")
    axs[1].set_xticks(np.arange(1, len(methods)+1)-0.5, methods, fontsize=16)
    axs[1].set_yticks(np.arange(1, df2.shape[0])-0.5, ["D" + str(i) for i in range(1, df2.shape[0])], fontsize=16, rotation=0, )
    axs[1].set_title(df2name)
    plt.tight_layout()
    plt.show()


def features_line_graph(all_values, datasets):
    datasets = [f'D{i}' for i in range(1, 21)]
    fig, axes = plt.subplots(4, 5, figsize=(20, 6))
    axes = axes.flatten()
    
    for i, values in enumerate(all_values):
        x = list(range(1, len(values) + 1))
        axes[i].plot(x, values, marker='o')
        axes[i].set_title(datasets[i], fontsize=18)
        if i >= 15:
            axes[i].set_xlabel('Number of Features', fontsize=18)
        if i % 5 == 0:
            axes[i].set_ylabel(r'$\overline{\Delta\theta}$', fontsize=18)
        axes[i].tick_params(axis='x', labelsize=14)
        axes[i].tick_params(axis='y', labelsize=14)
    axes[5].set_ylim(0.010)
    axes[15].set_xticks([2, 6, 10])
    axes[15].set_xticklabels([2, 6, 10], fontsize=14)
    plt.tight_layout()
    plt.show()