import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

my_format = mdates.DateFormatter('%d-%b-%y %H:%m')

def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), fontsize=10)

# Define plot function
def plot_actions(spot_price, action, opening_capacity=None, start=0, end=-1):
    """
    Notes: Plot where the algorithm charges and discharges
    ----------
    Parameters
    ----------
    spot_price       : dataframe with spot_price & forecast columns
    action           : discharge if value > 0, charge if value < 0
    opening_capacity : plot closing capacity if provided
    start            : start index (default=0)
    end              : end index (default=300)
    
    Returns
    -------
    plot with discharge and charge verticle lines
    """
    # Prepare bar chart data
    closing_capacity = opening_capacity.shift(-1)
    capacity_delta = closing_capacity - opening_capacity

    # Prepare bar chart colours
    condlist = [capacity_delta < 0, capacity_delta==0, capacity_delta > 0]
    choicelist = ['red', 'white', 'green']
    capacity_delta_colours = np.select(condlist, choicelist)

    bar_x_values = closing_capacity.index[start:end]
    bar_y_values = closing_capacity[start:end]

    fig, axs = plt.subplots(2, 1, figsize=(14,5), gridspec_kw={'height_ratios': [3, 1]})
    axs[0].plot(spot_price[start:end], label='Spot Price')
    axs[0].set_ylabel('Spot Price (AUD)', fontsize=10)
    axs[0].set_ylim(spot_price[start:end].min(), spot_price[start:end].max())
    axs[0].vlines(x = action[action > 0].index, ymin = -1000, ymax = 20000, linestyle = 'dotted', color = 'red', label = 'Discharge')
    axs[0].vlines(x = action[action < 0].index, ymin = -1000, ymax = 20000, linestyle = 'solid', color = 'green', label = 'Charge')
    legend_without_duplicate_labels(axs[0])
    axs[0].set_xlim(bar_x_values[0], bar_x_values[-1])

    axs[1].bar(x=bar_x_values,
                height=bar_y_values,
                color=capacity_delta_colours[start:end],
                align='edge',
                width=-0.005)
    axs[1].step(x=bar_x_values,
                y=bar_y_values,
                color='black')

    axs[1].set_ylabel('Capacity (MWh)', fontsize=10)
    axs[1].set_xlim(bar_x_values[0], bar_x_values[-1])
    axs[0].xaxis.set_major_formatter(my_format)
    axs[1].xaxis.set_major_formatter(my_format)
    plt.tight_layout()    

    return plt.show()