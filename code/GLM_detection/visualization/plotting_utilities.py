# Contains tools to aid in plotting and visualization
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.dates as md

def set_ticks(ax, tick_axis, num_ticks, format_str='%.2f', date_flag=False):
    """ Sets the number of ticks for the specified axis using the specified string format

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis to set the ticks for
    tick_axis : str
        The tick axis to set ('x', 'y')
    num_ticks : int
        Number of tick marks
    format_str : str
        The format used to display the tick mark labels
    date_flag : bool
        If True then plotting dates, so use date formatting

    """

    if tick_axis == 'x':
        xRange = ax.get_xlim()
        xticks = np.linspace(xRange[0], xRange[1], num_ticks)
        ax.set_xticks(xticks)
        if date_flag:
            ax.xaxis.set_major_formatter(md.DateFormatter(format_str))
        else:
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter(format_str))
    elif tick_axis == 'y':
        yRange = ax.get_ylim()
        yticks = np.linspace(yRange[0], yRange[1], num_ticks)
        ax.set_yticks(yticks)
        if date_flag:
            ax.yaxis.set_major_formatter(md.DateFormatter(format_str))
        else:
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(format_str))
    else:
        raise Exception('Unkown tick axis') 

    return
