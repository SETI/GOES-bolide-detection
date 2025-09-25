# improves on _v1 by 
# 1. saving data points corresponding to the PHT-identified line segements
# 2. adds toggle for ransac
# 3. implement an r2 filter applied to the ransac fits 
# test
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import RANSACRegressor
#from skimage.feature import canny
import copy
import os
#from haversine import haversine, Unit
#from skimage.transform import probabilistic_hough_line
#from skimage.draw import line
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import pickle
from bolide_detections import extract_groups_from_all_files

def normalize_image_data(df, pixel_x, pixel_y):
    """ Normalizes the GlmGroup spatial data and creates image (matrix) of data

    Parameters
    ----------
    df: [pandas DataFrame] dataframe with time, longitude, and latitude data
    pixel_x [float]: the number of pixels in the x direction
    pixel_y [float]: the number of pixels in the y direction

    Returns
    ----------
    df_norm: [pandas DataFrame] dataframe with time and normalized longitude and latitude data
    img: [numpy array of arrays] an img_dim by img_dim numpy array representing the image of the data
    glm_container: [list of list of ints] an img_dim by img_dim
                    list of list of 0's, used to keep track of object that spawned the data on the scatter plot
    lat_coords: [list of floats] List of latitude coordinates
    long_coords: [list of floats] List of longitude coordinates
    """

    df_norm = copy.deepcopy(df)

    # extract spatial data
    lat_coords = df_norm.latitudeDegreesNorth
    long_coords = df_norm.longitudeDegreesEast

    # Determine the range of the coordinates
    max_lat, min_lat = np.max(lat_coords), np.min(lat_coords)
    max_long, min_long = np.max(long_coords), np.min(long_coords)


    # Create an empty image
    img = np.zeros((pixel_y,pixel_x))
    # and an GlmGroups container ... this'll be used to keep track of the edges selected from the PHT
    glm_container = [[(0,0,0)] * pixel_x for _ in range(pixel_y)]

    # normalize
    df_norm['latitudeDegreesNorth'] = (df['latitudeDegreesNorth'] - min_lat) / (max_lat - min_lat) * (pixel_x-1)
    df_norm['longitudeDegreesEast'] = (df['longitudeDegreesEast'] - min_long) / (max_long - min_long) * (pixel_y - 1)

    # Map the data points to the image and the container
    for index, row_o in df.iterrows():
        # get normalized data
        row = df_norm.loc[index]
        # update image
        xi = int(row['latitudeDegreesNorth'])
        yi = int(row['longitudeDegreesEast'])
        img[yi, xi] = 1
        # store original data in identical container
        glm_container[yi][xi] = (row_o)

    return df_norm, img, glm_container, long_coords, lat_coords

def save_figure(fig, directory, filename):
    """Saves matplotlib figures"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(os.path.join(directory, filename))

def fix_coords(temporal_sort):
    """ TODO Ensures that longs are within [-180,180] and lats are within [-90,90]; modifies in place

    Parameters
    ----------
    temporal_sort [list GlmGroup objects]: a temporarly sorted list of GlmGroup objects that correspond to a PHT line segment;
    the first and last object represent the earliest and latest datapoints of a PHT line segement (i.e the first and last streak detection)

    Returns
    ----------
    None
    """
    for i in range(len(temporal_sort)):
        # fix lats
        if temporal_sort[i]['latitudeDegreesNorth'] < -90:
            temporal_sort[i] = (temporal_sort[i][0], temporal_sort[i][1], temporal_sort[i][2] + 180)
        elif temporal_sort[i]['latitudeDegreesNorth'] > 90:
            temporal_sort[i] = (temporal_sort[i][0], temporal_sort[i][1], temporal_sort[i][2] - 180)

        # fix longs
        if temporal_sort[i]['longitudeDegreesEast'] < -180:
            temporal_sort[i] = (temporal_sort[i][0], temporal_sort[i][1] + 360, temporal_sort[i][2])
        elif temporal_sort[i]['longitudeDegreesEast'] > 180:
            temporal_sort[i] = (temporal_sort[i][0], temporal_sort[i][1] - 360, temporal_sort[i][2])
    return

def get_statistics(line_seg_data):
    """ TODO """
    temporal_sort=sorted([x for x in line_seg_data], lambda x: x['time'])

    # streak duration
    tot_min = (temporal_sort[-1]['time'] - temporal_sort[0]['time']).total_seconds()/60

    # orientation
    fix_coords(temporal_sort)
    start = (float("{:.2f}".format(temporal_sort[0]['latitudeDegreesNorth'])), float("{:.2f}".format(temporal_sort[0]['longitudeDegreesEast'])))
    end = (float("{:.2f}".format(temporal_sort[-1]['latitudeDegreesNorth'])),  float("{:.2f}".format(temporal_sort[-1]['longitudeDegreesEast'])))

    # streak length
    cum_streak_length = 0
    curr=start
    for nxt in temporal_sort[1:-1]:
        nxt_coord=(nxt['latitudeDegreesNorth'],nxt['longitudeDegreesEast'])
        cum_streak_length+=haversine(curr, nxt_coord, unit = Unit.KILOMETERS)
        curr=nxt_coord

    print("Streak Duration [min]: ", float("{:.2f}".format(tot_min)))

    print("Starting Latitude:  ", start[0])
    print("Ending Latitude: ", end[0])

    print("Starting Longitude: ", start[1])
    print("Ending Longitude: ", end[1])

    print("Streak Length [km]", float("{:.2f}".format(cum_streak_length)))

    return

def to_pickle(files, output_path):
    """ extracts GlmGroup object data from files, stores it in a pandas dataframe, and saves it to a pickle file

    Parameters
    -----------
    files [list of strs]: list of absolulte paths to .nc files
    output_path [str]: the absolute path to the directory to save model outputs

    Returns
    -------
    df [pandas dataframe]: dataframe with all of the from GlmGroup objects in files
    """

    glmGroups,_ = extract_groups_from_all_files(files,eventIdListEnabled=False) # eventIdListEnabled=False speeds up function
    # create data collection object
    data = {
            'datasetFile':[],
            'id':[],
            'time':[],
            'latitudeDegreesNorth':[],
            'longitudeDegreesEast':[],
            'areaSquareKm':[],
            'energyJoules':[],
            'dataQuality':[],
            'eventIdList':[],
            'x':[],
            'y':[],
            }

    # loop through all of the data
    for obj in glmGroups:
        data['datasetFile'].append(obj.datasetFile)
        data['id'].append(obj.id)
        data['time'].append(obj.time)
        data['latitudeDegreesNorth'].append(obj.latitudeDegreesNorth)
        data['longitudeDegreesEast'].append(obj.longitudeDegreesEast)
        data['areaSquareKm'].append(obj.areaSquareKm)
        data['energyJoules'].append(obj.energyJoules)
        data['dataQuality'].append(obj.dataQuality)
        data['eventIdList'].append(obj.eventIdList)
        data['x'].append(obj.x)
        data['y'].append(obj.y)

    # Save the DataFrame to a pickle file
    df = pd.DataFrame(data=data)
    save_path = output_path+'dataframe.pkl'
    df.to_pickle(save_path)
    return df

def save_plotly_figure(fig, directory, filename):
    """Saves plotly figures"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the figure as an HTML file
    html_path = os.path.join(directory, f"{filename}.html")
    fig.write_html(html_path)
    return

def ransac(line_seg_data,r2_mse_cutoff,idx,display_plot,output_path,id,random_state=42):
    """ Runs RANSACRegressor raw GlmGroup data
        This function is intended to clean line segements detected by the PHT. As is the PHT gives us line segments but they may not
        be increasing in time. The RANSAC regressor figures out which subset of the data best best forms a line segment in 3d space

    Parameters
    ----------
    line_seg_data [list of GlmGroup]: list of raw GlmGroup data points
    r2_mse_cutoff [float]: a threshold for the r2 score, models less than this threshold will be discarded
    idx [int]: a sub batch id
    display_plot [bool]: if True displays plot
    output_path [string]: output directory
    id [int]: batch id

    Returns
    ----------
    filt_line_seg_data [list of GlmGroup]: list of raw GlmGroup data points, the inliers indentified by the ransac regression
    mse [float]: mean squared error
    r2 [float]: r squared score
    """
    # Ensure inputs are numpy arrays
    x=np.array([x['longitudeDegreesEast'] for x in line_seg_data])
    y=np.array([y['latitudeDegreesNorth'] for y in line_seg_data])
    z=np.array([z['time'].timestamp() for z in line_seg_data])
    z = (z - z.min())/ 60 # convert z to minutes

    # Create RANSACRegressor model
    ransac = RANSACRegressor(max_trials=1000)

    # Reshape input data
    X = np.column_stack((x, y))

    # Fit the model
    ransac.fit(X, z)

    # Predict using the fitted model
    z_pred = ransac.predict(X)

    # Get the inlier mask
    inlier_mask = ransac.inlier_mask_

    # Compute MSE for inliers
    mse = mean_squared_error(z[inlier_mask], z_pred[inlier_mask])
    print("MODEL MSE: ", mse)

    # Compute R-squared for inliers
    r2 = r2_score(z[inlier_mask], z_pred[inlier_mask])
    print("R-squared: ", r2)

    # Create a grid for plotting the best fit line
    # b/c we created an interpolated line between the p0 and p1 we are given that our points are monotonically increasing (or decreasing)
    # we need to determine whether we have an increase or decrease in spatial coordiantes to properly plot our line

    if x[0] <= x[-1]:
        x_range = np.linspace(x[inlier_mask].min(), x[inlier_mask].max(), 100)
    else:
        x_range = np.linspace(x[inlier_mask].max(), x[inlier_mask].min(), 100)

    if y[0] <= y[-1]:
        y_range = np.linspace(y[inlier_mask].min(), y[inlier_mask].max(), 100)
    else:
        y_range = np.linspace(y[inlier_mask].max(), y[inlier_mask].min(), 100)

    X_grid = np.column_stack((x_range, y_range))

    # Predict z values on the grid
    z_grid_pred = ransac.predict(X_grid)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{'type': 'scatter3d'}],[{'type': 'scatter'}]],
        subplot_titles=('3D Ground Trace', '2D Ground Trace')
    )

    # Plot 3D scatter plot with time as color gradient
    fig.add_trace(
        go.Scatter3d(
            x=x[inlier_mask], y=y[inlier_mask], z=z[inlier_mask], mode='markers', name=f"R2: {r2}",
            marker=dict(
                size=5,
                color=z[inlier_mask],  # Color by z (time)
                colorscale='Viridis',
                colorbar=None
            )
        ),
        row=1, col=1,
    )

    min_val = np.min(z[inlier_mask])
    max_val = np.max(z[inlier_mask])
    mid_val = (min_val + max_val) / 2
    tickvals = [min_val, mid_val, max_val]
    ticktext = [f"{min_val:.2f}", f"{mid_val:.2f}", f"{max_val:.2f}"]
    # Plot 2D scatter plot of Longitude vs Latitude with time as color gradient
    fig.add_trace(
        go.Scatter(
            x=x[inlier_mask], y=y[inlier_mask], mode='markers', name=f"MSE: {mse}",
            marker=dict(
                size=5,
                color=z[inlier_mask],  # Color by z (time)
                colorscale='Viridis',
                colorbar=dict(
                    title='Time [minutes]',
                    x=-0.3,  # Move color bar to the RHS
                    len=0.75,  # Length of the color bar
                    xpad=10,
                    ticks='outside',
                    tickvals=tickvals,
                    ticktext=ticktext
                )
            )
        ),
        row=2, col=1
    )

    # Plot RANSAC line in 3D
    fig.add_trace(
        go.Scatter3d(x=x_range, y=y_range, z=z_grid_pred, mode='lines', name='RANSAC 3D Prediction'),
        row=1, col=1
    )

    # Plot the projected RANSAC line in 2D
    fig.add_trace(
        go.Scatter(
            x=x_range, y=y_range, mode='lines', name='Best Fit Line',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"RANSAC REGRESSION {idx}",
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Time [min]'
        ),
        xaxis=dict(title='Longitude'),
        yaxis=dict(title='Latitude'),
        width=1000,
        height=1000
    )

    if display_plot and r2 >= r2_mse_cutoff:
        fig.show()
        save_plotly_figure(fig, output_path, f"{id}_{idx}_plotly_3D_ground_trace.png")
    filt_line_seg_data=[obj for obj, bool_val in zip(line_seg_data, inlier_mask) if bool_val]
    return filt_line_seg_data, mse, r2

def cphtr(df, output_path, config_dict, time_title, id):
        """ identifies streaks in data by applying a combined algorithmic approach uses canny edge detection, probablistic hough transform,
        and ransac regression

        Parameters
        ----------
        df: [pandas DataFrame] dataframe with time, longitude, and latitude data
        output_path [str]: the absolute path to the directory to save model outputs
        config_dict [dict]: a dictionary specifying the hyperparameters to the Probabalistic Hough Transform

        Returns
        ----------
        None
        """

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(time_title, fontsize=16)

        # extract model config_dict
        pht_threshold=config_dict['threshold'] # probabalistic hough transform (PHT) hyperparameter
        pht_line_length=config_dict['line_length'] # probabalistic hough transform (PHT) hyperparameter
        pht_line_gap=config_dict['line_gap'] # probabalistic hough transform (PHT) hyperparameter
        pixel_x=config_dict['pixel_x'] # number of pixels in x direction for the image passed to canny edge detection algo
        pixel_y=config_dict['pixel_y'] # number of pixels in x direction for the image passed to canny edge detection algo
        r2_mse_cutoff=config_dict['r2_mse_cutoff'] # combined r2 & mse cutoff filter # TODO
        display_plot=config_dict['display_plot'] # display matplotlib and plotly plots
        ransac_clean=config_dict['ransac_clean'] # run ransac to clean up PHT detected line segements

        # START PRE-PROCESS DATA
        #-------------------------------------------------------------------------
        # Normalize and scale the coordinates in the glmGroup data to fit within the image dimensions
        df_norm, img, glm_container, long_coords, lat_coords = normalize_image_data(df, pixel_x, pixel_y)

        # plot the raw data
        ax[0].scatter(long_coords, lat_coords, alpha=0.3, color='lightsteelblue')
        ax[0].set_title('Scatter Plot of Raw Data Points')
        ax[0].set_xlabel('Longitude')
        ax[0].set_ylabel('Latitude')
        ax[0].grid(True)


        # Perform edge detection on the image with the normalized data
        edges = canny(img.astype(float))
        ax[1].imshow(edges, cmap='hot')
        ax[1].set_title('Edge Image')
        ax[1].set_xticks([10, 100])
        ax[1].set_yticks([10, 100])

        # define the default PHT hyperparameter values: assume the variance in datapoints are no more than 20 km tranverse to the direction
        # of movement
        if pht_line_gap == 0.0:
            # Latitude: 1 deg = 110.574 km.
            # Longitude: 1 deg = 111.320*cos(latitude) km.
            pht_line_gap = 10

        # Plot Normalized data: extract normalized spatial data
        lat_scaled = df_norm.latitudeDegreesNorth
        long_scaled = df_norm.longitudeDegreesEast
        ax[2].scatter(long_scaled, lat_scaled, alpha=0.3,color='lightsteelblue')
        ax[2].set_xlim(min(long_scaled), max(long_scaled))
        ax[2].set_ylim(min(lat_scaled), max(lat_scaled))
        ax[2].set_title('Normalized Data')
        ax[2].set_xlabel('Normalized Longitude')
        ax[2].set_ylabel('Normalized Latitude')
        ax[2].grid(True)
        #-------------------------------------------------------------------------

        # Perform Probabilistic Hough Transform (PHT)
        #-------------------------------------------------------------------------
        line_segments = probabilistic_hough_line(edges, threshold=pht_threshold, line_length=pht_line_length, line_gap=pht_line_gap, rng=42)
        #-------------------------------------------------------------------------

        # check if any line segements detected
        #-------------------------------------------------------------------------
        pht_glm_groups = []
        if not line_segments:
            print("No PHT edge points detected")
            return True, []
        #-------------------------------------------------------------------------

        # plot the PHT detected interpolated line segments in pixel and raw data domain
        #-------------------------------------------------------------------------
        for (idx,seg) in enumerate(line_segments):
            line_seg_data = list()
            p0, p1 = seg # start and end point to PHT line segment (long, lat)

            # determine the coordinates of all points on the line segment connecting PHT detected edge points
            rr, cc = line(p0[0], p0[1], p1[0], p1[1])
            ax[2].plot((p0[1], p1[1]), (p0[0], p1[0]), 'r')

            # filter out the (0,0,0) points
            new_elements=pd.Series([id, idx], index=['batch_id', 'idx'])
            line_seg_data = [pd.concat([glm_container[c][r],new_elements]) for r, c in zip(rr, cc) if not np.array_equal(glm_container[c][r], (0, 0, 0))]

            # plot the interpolated line connecting line segment end points
            if len(line_seg_data) >= 2:
                start = line_seg_data[0]
                end = line_seg_data[-1]
                ax[0].plot(
                    (start['longitudeDegreesEast'], end['longitudeDegreesEast']),
                    (start['latitudeDegreesNorth'], end['latitudeDegreesNorth']),
                    'r')

            if ransac_clean:
                print("Running RANSAC cleaning.")
                filt_line_seg_data,mse,r2=ransac(line_seg_data,r2_mse_cutoff,idx,display_plot,output_path,id)
                if r2 >= r2_mse_cutoff:
                    line_seg_data=filt_line_seg_data
                else: continue
            # TODO
            # get line segement statistics
            # get_statistics(line_seg_data)
            pht_glm_groups.append(line_seg_data)
        #-------------------------------------------------------------------------

        # flatten array and temporal sort
        #-------------------------------------------------------------------------
        # there's the chance that our r2_mse_threshold will remove all pht detected line segmenets 
        # we need to incoporate error handling to handle this siutation 
        temporal_sort = sorted((item for sublist in pht_glm_groups for item in sublist), key=lambda x: x['time'])
        if len(temporal_sort) == 0: 
            print(f"No PHT edge points detected with given r2_mse_cutoff: {r2_mse_cutoff}.")
            return True, []
        #-------------------------------------------------------------------------

        # Plot the PHT line segment data in the raw data domain
        #-------------------------------------------------------------------------
        base_time = temporal_sort[0]['time'].timestamp()
        times_pht, longitudes_pht, latitudes_pht = [], [], []
        for obj in temporal_sort:
            times_pht.append((obj['time'].timestamp() - base_time) / 60)  # time array in minutes
            longitudes_pht.append(obj['longitudeDegreesEast'])
            latitudes_pht.append(obj['latitudeDegreesNorth'])

        scatter = ax[0].scatter(longitudes_pht, latitudes_pht, c=times_pht, cmap = plt.cm.viridis, s = 10)

        # # Add color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time [minutes]')

        save_figure(fig, output_path, f"{id}_matplotlib_2D_ground_trace.png")
        if display_plot:
            plt.show()
        #-------------------------------------------------------------------------
        return True, pht_glm_groups

def find_streaks(files, output_path, config_dict={
    'threshold':10,
    'line_length':120,
    'line_gap':0,
    'pixel_x':1372,
    'pixel_y':1300,
    'r2_mse_cutoff':0.00,
    'ransac_clean':True,
    'display_plot':False}):
    """ MASTER FUNCTION: Identifies near-field streaks in files, saves streak information to output directory

    Parameters
    -----------
    files [list of strs]: list of absolulte paths to .nc files
    output_path [str]: the absolute path to the directory to save model outputs
    config_dict [dict]: a dictionary specifying the hyperparameters to the Probabalistic Hough Transform

    Returns
    --------
    success [bool]: indicating completion of code
    """

    # download all of the data into a pickle file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df = to_pickle(files, output_path)
    df = df.sort_values(by='time')

    # Define the start and end time for the sliding window approach
    start_time = df['time'].min()
    end_time = df['time'].max()

    # Define the window size (1 hour) and the increment size (half an hour)
    window_size = pd.Timedelta(hours=1)
    increment_size = pd.Timedelta(minutes=30)

    # Initialize the current window start time
    current_window_start = start_time

    # Loop through the time range and access the data in each window
    success,all_line_seg=[],[]
    id=1
    while current_window_start + window_size <= end_time:
        # Define the window end time
        current_window_end = current_window_start + window_size

        # Extract the data for the current window
        window_data = df[(df['time'] >= current_window_start) & (df['time'] < current_window_end)]
        # Process the window_data as needed
        print(f"Window from {current_window_start} to {current_window_end}")

        # Extract hour and minute from the minimum time
        min_hour = current_window_start.hour
        min_minute = current_window_start.minute

        # Extract hour and minute from the maximum time
        max_hour = current_window_end.hour
        max_minute = current_window_end.minute
        time_title = f"{min_hour}:{min_minute} - {max_hour}:{max_minute}"

        # run analysis and save results
        res_bool, pht_line_segs = cphtr(window_data, output_path, config_dict, time_title, id)
        for line_seg in pht_line_segs:
            all_line_seg.append(line_seg)
        success.append(res_bool)

        # Increment the window start time
        current_window_start += increment_size
        id+=1

    # save the all of the line seg data
    save_path=output_path+'all_line_seg_data.pkl'
    with open(save_path, 'wb') as file:
        pickle.dump(all_line_seg, file)
    print("Number of PHT Line Detections: ",sum(success))
    return True in success

