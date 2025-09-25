#!/usr/bin/env python
import argparse
import pickle
import os
import csv
import math
import datetime

# This may not be needed but keeping here for reference
#import matplotlib
#matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from tkinter import *
from tkinter.messagebox import askokcancel
from tkinter import simpledialog

import bolide_detections as bd
import bolide_dispositions as bDisposition
import plot_bolide_detections as pbd
import scatter_lasso_tool as lassoTool

DEFAULT_BELIEF = None #0.5

def belief_to_score(b, minScore, maxScore):
    return round(b * float(maxScore - minScore)) + minScore

def score_to_belief(score, minScore, maxScore):
    return float(score - minScore) / float(maxScore - minScore)


class WrappingLabel(Label):
    # a type of Label that automatically adjusts the wrap to the size
    def __init__(self, master=None, **kwargs):
        Label.__init__(self, master, **kwargs)
        self.bind('<Configure>', lambda e: self.config(wraplength=self.winfo_width()))


# *****************************************************************************
# GlmLabelingGui
# --------------
# A GUI for interactively labeling bolide detections.
#
# CLASS ATTRIBUTES
#
#     TITLE_BG_COLOR
#     SLIDER_BG_COLOR
#     BUTTON_BG_COLOR
#
#     TITLE_FRAME_NUMBER
#     FIGURE_FRAME_NUMBER
#     SLIDER_FRAME_NUMBER
#     BUTTON_FRAME_NUMBER
#
# INSTANCE ATTRIBUTES
#   sprofileList
#   root
#   outbase
#   score
#   index           -- [int] the currently displayed bolide detection
#   infile
#   progressString
#   commentString
#   frames
#   userName
#   userExpertise
#   minSCore
#   maxScore
#   fig             -- The persistent figure handle
#   toolbar         -- The persistent toolbar
#   selector        -- The persistent lasso selector from pbd.plot_detections
#   profileList     -- [bolide_detections.BolideDispositionProfile array] the collection of bolide detections and
#                       dispositions
# *****************************************************************************
class GlmLabelingGui:

    TITLE_BG_COLOR  = '#D9D9D9'
    SLIDER_BG_COLOR = '#EEE8AA' # '#48D1CC' # '#F0E68C'
    BUTTON_BG_COLOR = '#D9D9D9'
    FONT_SIZE_STR = '14'

    def __init__(self, root, profileList, outbase, nLevels, useSlider, embedFigure, fig):

        self.embedFigure = embedFigure
        if self.embedFigure:
            self.NUM_FRAMES = 5
            self.TITLE_FRAME_NUMBER = 0
            self.FIGURE_FRAME_NUMBER = 1
            self.COMMENT_FRAME_NUMBER = 2
            self.SLIDER_FRAME_NUMBER = 3
            self.BUTTON_FRAME_NUMBER = 4
        else:
            self.NUM_FRAMES = 4
            self.TITLE_FRAME_NUMBER = 0
            self.COMMENT_FRAME_NUMBER = 1
            self.SLIDER_FRAME_NUMBER = 2
            self.BUTTON_FRAME_NUMBER = 3

        self.profileList = profileList
        self.root = root
        self.outbase = outbase
        self.score = IntVar()
        self.index = 0
        self.infile = StringVar()
        self.progressString = StringVar()
        self.frames = []
        self.commentString = StringVar()
        self.userName = StringVar()
        self.userExpertise = DoubleVar()
        self.minScore = -math.floor(nLevels/2)
        self.maxScore = math.floor(nLevels/2)
        self.fig = fig
        self.toolbar = []
        self.selector = []

        # ---------------------------------------------------------------------
        # Create the frames.
        for i in range(self.NUM_FRAMES):
            self.frames.append(Frame(root))
            self.frames[i].pack(side=TOP, expand=YES, fill=BOTH)

        self.frames[self.TITLE_FRAME_NUMBER ].config( background = self.TITLE_BG_COLOR )
        self.frames[self.SLIDER_FRAME_NUMBER].config( background = self.SLIDER_BG_COLOR )
        self.frames[self.COMMENT_FRAME_NUMBER].config( background = self.TITLE_BG_COLOR )
        self.frames[self.BUTTON_FRAME_NUMBER].config( background = self.BUTTON_BG_COLOR )

        # ---------------------------------------------------------------------
        # Add user information to the user frame.
        Label(self.frames[self.TITLE_FRAME_NUMBER], text='Name:', font=('Ariel',self.FONT_SIZE_STR,'bold'), background = self.TITLE_BG_COLOR ).grid(row=0, column=0, sticky='nw' )
        Label(self.frames[self.TITLE_FRAME_NUMBER], textvariable=self.userName, font=('Ariel',self.FONT_SIZE_STR), background = self.TITLE_BG_COLOR ).grid(row=0, column=1, sticky='nw')
        Label(self.frames[self.TITLE_FRAME_NUMBER], text='Expertise:', font=('Ariel',self.FONT_SIZE_STR,'bold'), background = self.TITLE_BG_COLOR ).grid(row=1, column=0, sticky='nw')
        Label(self.frames[self.TITLE_FRAME_NUMBER], textvariable=self.userExpertise, font=('Ariel',self.FONT_SIZE_STR), background = self.TITLE_BG_COLOR ).grid(row=1, column=1, columnspan=2, sticky='nw')

        # Add labels to display progress and data file names.
        self.progressLabel = Label(self.frames[self.TITLE_FRAME_NUMBER],
            textvariable=self.progressString, font=('Ariel',self.FONT_SIZE_STR,'bold'),
            background = self.TITLE_BG_COLOR ).grid(row=2, column=0, sticky='nw')
        Entry( self.frames[self.TITLE_FRAME_NUMBER],
                textvariable=self.infile, width=len(self.infile.get()),
                relief='flat', state='readonly',
                background = self.TITLE_BG_COLOR, insertbackground=self.TITLE_BG_COLOR,
                readonlybackground='white', fg='black').grid(row=2, column=1, sticky='nw')
        self.update_progress_string()
        self.set_file_name()

        # ---------------------------------------------------------------------
        # Comment frame
        Label(self.frames[self.COMMENT_FRAME_NUMBER],
            text='Comments:', font=('Ariel',14, 'bold'),
            background = self.TITLE_BG_COLOR ).grid(row=0, column=0)
        Button (self.frames[self.COMMENT_FRAME_NUMBER], text="Edit",
                command=self.edit_comment, height=2).grid(row=1, column=0)
        self.commentEntry = Entry( self.frames[self.COMMENT_FRAME_NUMBER],
                textvariable=self.commentString, width=max(80, len(self.commentString.get())),
                relief='flat', state='readonly',
                background = self.TITLE_BG_COLOR, insertbackground=self.TITLE_BG_COLOR,
                readonlybackground='white', fg='black')
        self.commentEntry.grid(row=1, column=1, )
        Button (self.frames[self.COMMENT_FRAME_NUMBER], text="Glint", fg='red',
                command=self.add_glint_comment, height=2).grid(row=1, column=3)

        self.frames[self.COMMENT_FRAME_NUMBER].grid_columnconfigure(1, weight=1)
        self.frames[self.COMMENT_FRAME_NUMBER].grid_columnconfigure(2, weight=2)
        self.frames[self.COMMENT_FRAME_NUMBER].grid_columnconfigure(3, weight=1)

        # ---------------------------------------------------------------------
        # Add interactive slider or radio buttons.
        if useSlider :
            Label(self.frames[self.SLIDER_FRAME_NUMBER],
                  text='Not\nBolide', anchor=CENTER, font=('Ariel', self.FONT_SIZE_STR, 'bold'),
                  background=self.SLIDER_BG_COLOR).pack(side=LEFT, expand=YES, fill=X)
            Scale(self.frames[self.SLIDER_FRAME_NUMBER],
                            variable=self.score,
                            from_=self.minScore, to=self.maxScore,
                            command=self.on_move,
                            background = self.SLIDER_BG_COLOR,
                            orient=HORIZONTAL).pack(side=LEFT, expand=YES, fill=X)
            Label(self.frames[self.SLIDER_FRAME_NUMBER], background = self.SLIDER_BG_COLOR,
                  text='\nBolide', anchor=CENTER, font=('Ariel',self.FONT_SIZE_STR,'bold')).pack(side=LEFT, expand=YES, fill=X)
        else:
            # Note the use of the 'weight' option with the grid geometry manager. This assigns relative priorities
            # to the widgets when filling space within the frame.
            Label(self.frames[self.SLIDER_FRAME_NUMBER],
                  text='Not\nBolide', anchor=CENTER, font=('Ariel', self.FONT_SIZE_STR, 'bold'),
                  background=self.SLIDER_BG_COLOR).grid(row=0, column=0, sticky='WE')
            c = 1
            for val in range(self.minScore, self.maxScore + 1) :
                c = c + 1
                Radiobutton(self.frames[self.SLIDER_FRAME_NUMBER],
                      variable=self.score,
                      value=val,
                      command=self.on_select,
                      anchor=CENTER,
                      background=self.SLIDER_BG_COLOR).grid(row=0, column=c, sticky='WE')

            Label(self.frames[self.SLIDER_FRAME_NUMBER], background=self.SLIDER_BG_COLOR,
                      text='\nBolide', anchor=CENTER, font=('Ariel', self.FONT_SIZE_STR, 'bold')).grid(row=0, column=c+1, sticky='WE'  )

            self.frames[self.SLIDER_FRAME_NUMBER].grid_columnconfigure(0, weight=1)
            col_count, row_count = self.frames[self.SLIDER_FRAME_NUMBER].grid_size()
            for col in range(1, col_count):
                self.frames[self.SLIDER_FRAME_NUMBER].grid_columnconfigure(col, weight=3)
            self.frames[self.SLIDER_FRAME_NUMBER].grid_columnconfigure(col_count-1, weight=1)

        # ---------------------------------------------------------------------
        # Insert control buttons.
        Button (self.frames[self.BUTTON_FRAME_NUMBER], text="< Prev",
                command=self.prev_figure).pack(side=LEFT, expand=YES)
        Button (self.frames[self.BUTTON_FRAME_NUMBER], text=" Next >",
                command=self.next_figure).pack(side=LEFT, expand=YES)
        Button (self.frames[self.BUTTON_FRAME_NUMBER], text="Save",
                command=self.save).pack(side=LEFT, expand=YES)
        Button (self.frames[self.BUTTON_FRAME_NUMBER], text="Quit",
                command=self.quit).pack(side=LEFT, expand=YES)

        # ---------------------------------------------------------------------
        # Key bindings.
        root.bind('<Left>',  self.prev_figure)
        root.bind('<Right>', self.next_figure)
        root.bind('s', self.save)
        root.bind('q', self.quit)
        root.bind('e', self.edit_comment)

        # ---------------------------------------------------------------------
        # Initialize the canvas if embedding a figure
        if (self.embedFigure):
            self.frames[self.FIGURE_FRAME_NUMBER].canvas = \
                FigureCanvasTkAgg(Figure(), master=self.frames[self.FIGURE_FRAME_NUMBER])

        # ---------------------------------------------------------------------
        # Refresh 

        self.refresh()

    # Note the dummy 'event' argument to allow tkinter key binding to this method.
    def add_glint_comment(self, event=None):
        s = self.profileList[self.index].humanOpinions[-1].comments
        if isinstance(s, str) :
            sNew = s + '(GLINT)'
        else:
            sNew = '(GLINT)'
        self.commentString.set(sNew)
        self.set_comment()
        self.set_score(self.minScore) # Since we're sure this is glint, set the score to the lowest value.

    # Given a score, set both the score variable and human opinion.
    def set_score(self, score):
        self.score.set(score)
        self.profileList[self.index].humanOpinions[-1].belief = score_to_belief(score, self.minScore, self.maxScore)
        self.profileList[self.index].humanOpinions[-1].time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Given a belief, set both the human opinion and set the score variable.
    def set_belief(self, belief):
        self.score.set(belief_to_score(belief, self.minScore, self.maxScore))
        self.profileList[self.index].humanOpinions[-1].belief = belief

    # Note the dummy 'event' argument to allow tkinter key binding to this method.
    def edit_comment(self, event=None):
        self.commentEntry.config(state=NORMAL)
        s = self.profileList[self.index].humanOpinions[-1].comments
        sNew = simpledialog.askstring('Edit Comment', prompt='', initialvalue=s )
        if sNew != None and sNew != s :
            self.profileList[self.index].humanOpinions[-1].comments = sNew
            self.commentString.set(sNew)
            self.set_comment()
        self.commentEntry.config(state='readonly')

    def update_comment_string(self):
        self.commentString.set(self.profileList[self.index].humanOpinions[-1].comments)

    def set_comment(self):
        self.profileList[self.index].humanOpinions[-1].comments = self.commentString.get()

    def update_progress_string(self):
        self.progressString.set( 'GLM File ({}/{})'.format(self.index + 1, len(self.profileList) ) )

    def set_file_name(self):
        self.infile.set(os.path.basename(self.profileList[self.index].bolideDetection.filePathList[0]))

    def refresh(self):
        self.update_progress_string()
        self.update_comment_string()
        self.set_file_name()
        self.plot()
        self.userName.set(self.profileList[self.index].humanOpinions[-1].name)
        self.userExpertise.set(self.profileList[self.index].humanOpinions[-1].expertise)

        # Here we handle the case where no belief has been entered. No button will be highlighted and no value will
        # be written to the CSV file.
        belief = self.profileList[self.index].humanOpinions[-1].belief
        if belief != None :
            self.score.set( int(belief_to_score(self.profileList[self.index].humanOpinions[-1].belief, self.minScore, self.maxScore)) )
        else:
            self.score.set(None)


    # Note the dummy 'event' argument to allow tkinter key binding to this method.
    def prev_figure(self, event=None):
        self.index -= 1
        if self.index < 0 :
            self.index = 0
        self.refresh()

    # Note the dummy 'event' argument to allow tkinter key binding to this method.
    def next_figure(self, event=None):
        self.index += 1
        numDetections = len(self.profileList)
        if self.index > numDetections - 1 :
            self.index = numDetections - 1
        self.refresh()

    # Note the dummy 'event' argument to allow tkinter key binding to this method.
    def save(self, event=None):
        self.refresh()
        self.write_profiles()
        self.write_csv_file()

    def write_profiles(self):
        outfile = self.outbase + '.p'
        try:
            with open(outfile, 'wb') as fp:
                print('Saving data to {}'.format(outfile))
                pickle.dump(self.profileList, fp)
        except:
            sys.exit('Could not write to file {}.'.format(outfile))
        fp.close()

    def write_csv_file(self):
        outfile = self.outbase + '.csv'
        try:
            with open(outfile, 'w') as fp:
                print('Saving data to {}'.format(outfile))
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(['CONFIDENCE', 'LABEL_TIME', 'START_TIME', 'END_TIME', 'LAT_DEGREES', 'LON_DEGREES', 'JOULES', 'COMMENTS', 'NAME', 'EXPERTISE', 'FILE_NAMES', 'ID_TYPE', 'ID_LIST'])
                for profile in self.profileList :
                    row = self.build_csv_row(profile)
                    wr.writerow(row)
        except:
            sys.exit('Could not write to file {}.'.format(outfile))
        fp.close()

    def build_csv_row(self, profile):
        confidence = profile.humanOpinions[-1].belief
        labelTime = profile.humanOpinions[-1].time
        startTime, endTime = profile.bolideDetection.get_time_interval()
        lat, lon = profile.bolideDetection.get_average_group_lat_lon()
        energy = profile.bolideDetection.get_total_energy()
        filePathList = profile.bolideDetection.filePathList
        name = profile.humanOpinions[-1].name
        expertise = profile.humanOpinions[-1].expertise
        commentStr = profile.humanOpinions[-1].comments

        # Choose the more compact representation.
        if len(profile.bolideDetection.groupList) < 1 :
            typeStr = 'event_id'
            ids = [event.id for event in profile.bolideDetection.eventList]
        else:
            typeStr = 'group_id'
            ids = [group.id for group in profile.bolideDetection.groupList]

        row = [confidence] + [labelTime] + [startTime] + [endTime] + [lat] + [lon] + [energy] + \
              [commentStr] + [name] + [expertise] + [filePathList] + [typeStr] + [ids]

        return row

    # Note the dummy 'event' argument to allow tkinter key binding to this method.
    def quit(self, event=None):
        ans = askokcancel('Verify exit', "Are you sure you want to quit?")
        if ans == True :
            sys.exit('Exiting ...')

    def on_move(self, value):
        b = score_to_belief(int(value), self.minScore, self.maxScore)
        self.profileList[self.index].humanOpinions[-1].belief = b
        self.profileList[self.index].humanOpinions[-1].time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def on_select(self):
        b = score_to_belief(int(self.score.get()), self.minScore, self.maxScore)
        self.profileList[self.index].humanOpinions[-1].belief = b
        self.profileList[self.index].humanOpinions[-1].time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def plot(self):

        # The figure is persistent (self.fig) and created once at the beginning.

        # If we are embeding the figure in the TK GUI window then certain interactive figure functionality is not
        # available
        if self.embedFigure:
            # Destroy the existing canvas before creating a new one.
            self.frames[self.FIGURE_FRAME_NUMBER].canvas.get_tk_widget().destroy()
            
            # Generate the figure from the detection object.
            # Do NOT enable interactive or a seperate window will pop up.
            [self.fig, self.selector] = pbd.plot_detections(self.profileList[self.index].bolideDetection, showDataFileName=False,
                    interactiveEnabled=False, figure=self.fig)
            
            # Embed the figure in a new canvas.
            self.frames[self.FIGURE_FRAME_NUMBER].canvas = FigureCanvasTkAgg(self.fig, master=self.frames[self.FIGURE_FRAME_NUMBER])
            self.frames[self.FIGURE_FRAME_NUMBER].canvas.get_tk_widget().pack(expand=YES, fill=BOTH)
            self.frames[self.FIGURE_FRAME_NUMBER].canvas.draw()
            
            # Add a toolbar so that we can manipulate the figure
            # We want the new toolbar to overwrite the old. Otherwise, a new one is created above the old. 
            # So, we make it persistent and "destroy" it before refreshing (But only if one has already been created.)
            if (self.toolbar != []):
                self.toolbar.destroy()
            
            self.toolbar = NavigationToolbar2Tk(self.frames[self.FIGURE_FRAME_NUMBER].canvas, root)
            self.toolbar.update()
            self.frames[self.FIGURE_FRAME_NUMBER].canvas.get_tk_widget().pack(expand=NO, fill=BOTH)
        else:
            # Generate the figure from the detection object.
            # Do NOT enable interactive or a seperate window will pop up.
            [self.fig, self.selector] = pbd.plot_detections(self.profileList[self.index].bolideDetection, showDataFileName=False,
                    interactiveEnabled=True, figure=self.fig)
            


# *****************************************************************************
# Parse an argument list.
#
# INPUTS
#     arg_list : A list of strings, each containing a command line argument.
#                NOTE that the first element of this list should NOT be the
#                program file name. Instead of passing sys.argv, pass
#                arg_list = sys.argv[1:]
#
# OUTPUTS
#     args     : A Namespace containing the extracted arguments.
# *****************************************************************************
def parse_arguments(arg_list):
    parser = argparse.ArgumentParser(description='Label bolide detections.')
    parser.add_argument('infile', metavar='infile', type=str, nargs=1,
                        help='Input file name')
    parser.add_argument('outbase', metavar='outbase', type=str, nargs=1,
                        help='Output file base name (extensions are ignored)')
    parser.add_argument('--name', '-n', dest='name', type=str, default='',
                        help='The name of the human user (default: None)')
    parser.add_argument('--expertise', '-e', dest='expertise', type=float, default=0,
                        help='The user\'s level of expertise in the interval [0,1] (default: None)')
    parser.add_argument('--csv', '-c', dest='csv', action='store_true',
                        help='Indicates infile is a CSV file (default: False)')
    parser.add_argument('--scale_levels', '-l', dest='levels', type=int, default=3,
                        help='The integer number of confidence levels (default: 3)')
    parser.add_argument('--slider', '-s', dest='useSlider', action='store_true',
                        help='Use a slider instead of radio buttons (default: False)')
    parser.add_argument('--embedFigure', '-ef', dest='embedFigure', action='store_true',
                        help='Embed the figure in TK GUI (default: False, seperate window)')

    args = parser.parse_args(arg_list)

    return args


# *****************************************************************************
# Run the labeling GUI.
# *****************************************************************************
if __name__ == "__main__":

    # Make sure we're running Python 3
    if sys.version_info[0] < 3:
        raise Exception("Python 3.0 or higher is required")

    args = parse_arguments(sys.argv[1:])

    infile      = args.infile[0]
    outbase     = args.outbase[0]
    userName    = args.name
    expertise   = args.expertise
    readCsv     = args.csv
    nLevels     = args.levels
    useSlider   = args.useSlider
    embedFigure = args.embedFigure

    # Remove any extension from the output file name.
    outbase = os.path.splitext(outbase)[0]

    # Read the detection file and create the list of detection summary objects.
    if readCsv:
        print('Constructing bolideDetection objects from CSV records ...')
        detectionRecordList = bd.read_bolide_detection_records(infile)
        objList = []
        for record in detectionRecordList:
            objList.append(bd.bolideDetection.fromGlmDataFiles(record))
    else:
        objList = bd.unpickle_bolide_detections(infile)

    if not isinstance(objList, list) :
        sys.exit('Unable to construct object list from file {}. '.format(infile))

    # Now that we have loaded in the bolide detection, construct the candidate profile objects
    if isinstance(objList[0], bd.bolideDetection) :
        profileList = []
        for detection in objList:
            profileList.append(bDisposition.BolideDispositionProfile(detection,
                humanOpinions=[bDisposition.HumanOpinion(DEFAULT_BELIEF, userName, expertise)]))
    elif isinstance(objList[0], bDisposition.BolideDispositionProfile) :
        # This is if we are loading in a data set which has already been dispositioned
        profileList = objList
        # for profile in profileList:
        #     profile.humanOpinions.append(bDisposition.HumanOpinion(0.5, userName, expertise))
    else:
        sys.exit('Unknown object type in file {}. '.format(infile))

    root = Tk()
    # Need to make TK GUI larger to accomodate embedded figure
    if embedFigure:
        root.geometry('900x800')  # Width x Height
    else:
        root.geometry('900x250')  # Width x Height

    fig = plt.figure() # Create the persistent figure
    start = GlmLabelingGui (root, profileList, outbase, nLevels, useSlider, embedFigure, fig)
    root.mainloop()

    GlmLabelingGui.selector.disconnect()
    plt.close(fig)

# ************************************ EOF ************************************
