#************************************************************************************************************
# scatter_lasso_too.py
#
# This tool allows one to "lasso" a selection of points in a scatter plot to select them.
#
# LassoSelector tends to black out the axis when selecting with multiple axis. So, using RectangleSelector
# instead.
#
# See __main__ function for a demo.
#
#************************************************************************************************************


import numpy as np

from matplotlib.widgets import LassoSelector, RectangleSelector
from matplotlib.path import Path


class SelectFromCollection(object):
    """
    From: https://matplotlib.org/gallery/widgets/lasso_selector_demo_sgskip.html#sphx-glr-gallery-widgets-lasso-selector-demo-sgskip-py

    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    If multiple axes are passed then all axes will be linked. A lasso around one will cause the saem points to be
    highlighted in the other axes. This assumes the data plotted in each axes is for the same data set and in the same
    sequential order.
    Note: Axes and collection lists must be equal length

    Parameters
    ----------
    axes        : :class list:`~matplotlib.axes.Axes` Axes to interact with.

    collection: :class list:`matplotlib.collections.Collection` subclass
                     Collection you want to select from.

    alpha_other : 0 <= float <= 1
                    To highlight a selection, this tool sets all selected points to an
                    alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, axes, collection, alpha_other=0.3):

        if (len(axes) != len(collection)):
            raise Exception("<axes> and <collection> must be the same legth")

        self.nAxes = len(axes)

        self.canvas = axes[0].figure.canvas # There is only one canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = [col.get_offsets() for col in collection]
        self.Npts = [len(xys) for xys in self.xys]

        if (not np.all([Npts == self.Npts[0] for Npts in self.Npts[1:]])):
            raise Exception("The number of points in all collections must be equal")
        else:
            self.Npts = self.Npts[0]

        # Get the face colors so we know what to set them to after the lasso
        # Ensure that we have separate colors for each object so that we can manipulate each point's color seperately
        # Before we can get the face/egde colors we first need to draw the figure to generate the data to get (Guess how
        # long it took for me to discover this!)
        self.canvas.draw()
        self.fcSave = [col.get_facecolors() for col in self.collection]
        for iAxis in np.arange(self.nAxes):
            if (len(self.fcSave[iAxis]) == 0):
                raise ValueError('Collection must have a facecolor')
            if len(self.fcSave[iAxis]) == 1:
                self.fcSave[iAxis] = np.tile(self.fcSave[iAxis], (self.Npts, 1))

       #self.lasso  = [LassoSelector(axis, onselect=self.onselect) for axis in axes]

     #  iAxis = 1
     #  # LassoSelector tends to black out the axis when selecting
     # #self.selectorObj  = LassoSelector(axes[iAxis], onselect=lambda verts : self.onselect(iAxis, verts))

        # TODO: Get the for loop working. For some reason instantiating these objects in this for-loop results in only the last
        # axis to be active in section.
     #  self.selectorObj = []
     #  for iAxis in np.arange(self.nAxes):
     #      self.selectorObj.append(RectangleSelector(axes[iAxis], onselect=lambda eclick, erelease : self.onselect(iAxis, eclick, erelease)))

        self.selectorObj0  = RectangleSelector(axes[0], onselect=lambda eclick, erelease : self.onselect(0, eclick, erelease))
        self.selectorObj1  = RectangleSelector(axes[1], onselect=lambda eclick, erelease : self.onselect(1, eclick, erelease))
        self.selectorObj2  = RectangleSelector(axes[2], onselect=lambda eclick, erelease : self.onselect(2, eclick, erelease))
        self.selectorObj3  = RectangleSelector(axes[3], onselect=lambda eclick, erelease : self.onselect(3, eclick, erelease))


        self.ind    = [[] for i in np.arange(self.nAxes)]

        pass

    #*********************************************************************************************************
    # This one is for RectangleSelector
    # When we select points, perform these actions
    #
    # This method will determine which datums were selected int eh amster Axis and then highlight those datums on all
    # Axis.
    #
    # Inputs:
    #   iAxisMaster -- [int] index of the axes we selected the points from
    #   eclick      -- [] Information on the initial mouse click 
    #   erelease      -- [] Information on the mouse unclick (release of button) 
    #
    #*********************************************************************************************************
    def onselect(self, iAxisMaster, eclick, erelease):

        # Select the datum indices to highlight
        path = Path([[eclick.xdata,   eclick.ydata],
                     [eclick.xdata,   erelease.ydata],
                     [erelease.xdata, erelease.ydata],
                     [erelease.xdata, eclick.ydata]])
        self.ind[iAxisMaster] = np.nonzero(path.contains_points(self.xys[iAxisMaster]))[0]

        for iAxis in np.arange(self.nAxes):
            # Use .copy so that we make an indepdendent copy of the object handle.
            # Otherwise, as we change fc, self.fcSave will also change!
            fc = self.fcSave[iAxis].copy()
            
            # If points are selected then highlight them
            # Otherwise reset the figure
            if (len(self.ind[iAxisMaster]) > 0):
                # First set all points face to alpha_other
                fc[:, -1] = self.alpha_other
                # Set the lasso'd points to normal
                fc[self.ind[iAxisMaster], -1] = 1
                self.collection[iAxis].set_facecolors(fc)
            else:
                # Set all points to normal
                self.collection[iAxis].set_facecolors(self.fcSave[iAxis])

        self.canvas.draw_idle()

    #*********************************************************************************************************
    # This one is for LassoSelector
    # When we select, perform these actions
  # def onselect(self, iAxisMaster, verts):

  #     # Select the datum indices to highlight
  #     path = Path(verts)
  #     self.ind[iAxisMaster] = np.nonzero(path.contains_points(self.xys[iAxisMaster]))[0]

  #     for iAxis in np.arange(self.nAxes):
  #         # Use .copy so that we make an indepdendent copy of the object handle.
  #         # Otherwise, as we change fc, self.fcSave will also change!
  #         fc = self.fcSave[iAxis].copy()
  #         
  #         # If points are selected then highlight them
  #         # Otherwise reset the figure
  #         if (len(self.ind[iAxisMaster]) > 0):
  #             # First set all points face to alpha_other
  #             fc[:, -1] = self.alpha_other
  #             # Set the lasso'd points to normal
  #             fc[self.ind[iAxisMaster], -1] = 1
  #             self.collection[iAxis].set_facecolors(fc)
  #         else:
  #             # Set all points to normal
  #             self.collection[iAxis].set_facecolors(self.fcSave[iAxis])

  #     self.canvas.draw_idle()

    #*********************************************************************************************************
    # Do this once we are finished to reset
    def disconnect(self):
        self.selectorObj0.disconnect_events()
        self.selectorObj1.disconnect_events()
        self.selectorObj2.disconnect_events()
        self.selectorObj3.disconnect_events()
        # Set all points to normal
        for iAxis in np.arange(self.nAxes):
            self.collection[iAxis].set_facecolors(self.fcSave[iAxis])
        self.canvas.draw_idle()

#************************************************************************************************************

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    data = np.random.rand(100, 2)

    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    selector = SelectFromCollection(ax, pts)

    def accept(event):
        if event.key == "enter":
            print("Selected points:")
            print(selector.xys[selector.ind])
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")

    plt.show()
