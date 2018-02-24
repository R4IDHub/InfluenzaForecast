"""
@author: Anton Sporrer, anton.sporrer@yahoo.com
"""

# Basic imports
import os
from collections import OrderedDict, defaultdict
import re
from datetime import datetime
from isoweek import Week
import pandas as pd
import numpy as np

# Providing Data Source and Data Formatting Functionality:
from Data.DataFormatting import DataFrameProvider
import Data.DataFormatting as DataFormatting

# Storing and fetching data
import dill as pickle

# Pipeline
from sklearn.pipeline import Pipeline
# Scaling
from sklearn.preprocessing import StandardScaler
# Kernel Density Estimation for Visualization
from sklearn.neighbors import KernelDensity
# Classification
from sklearn.neural_network import MLPClassifier
# Evaluation and Metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, log_loss, fbeta_score, roc_auc_score, accuracy_score, recall_score, \
    precision_score, confusion_matrix

# Visualization
# from bokeh.plotting import *
from bokeh.plotting import figure, show
from bokeh.embed import components
from bokeh.transform import linear_cmap
from bokeh.palettes import Category20b, Spectral6
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    FixedTicker,
    ColorBar,
    DatetimeTickFormatter,
    LogColorMapper,
    LabelSet,
    Range1d,
    LinearAxis
)

import shapefile

import matplotlib.pyplot as plt
import seaborn as sns

all_states_list = ['Baden-Wuerttemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg', 'Hessen',
                   'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland',
                   'Sachsen', 'Sachsen-Anhalt', 'Schleswig-Holstein', 'Thueringen']

"""
This module can be separated into two parts. 

The first part "I.) Statistical and Visual Exploration" provides the code
for the statistical and visual exploration. Again this part is divided into two subsections. First a subsection for 
generating the statistics and second a subsection for visualization.

The second part "II.) Classification" is concerned with the influenza forecast or more precisely with the classification associated to the 
forecast problem. The class ForecastProvider does expose the methods to train a classification model and predict whether
a certain threshold of reported number of influenza cases is crossed. Further the second part provides function for 
saving, transforming and visualizing the data provided by the ForecastProvider. The transformed data is used for the 
animation on the influenza forecast website for instance.
"""

########################################
########################################
# I.) Statistical and Visual Exploration
########################################
########################################

#############################
# I.1.) Generating Statistics
#############################

def get_first_length_count_df(first_list, length_list):
    """
    This function counts how often a certain combinations of (start week of wave, wave length in weeks) occurs and
    returns a data frame containing this information.

    :param first_list: A list, containing the first week of a waves.
    :param length_list: A list, containing the wave lengths.
    :return: A pandas.DataFrame, containing the count of how often a certain combinations of (start week of wave, wave
    length in weeks) occurred
    """
    count_dict = defaultdict(lambda: 0)
    for index in range(len(first_list)):
        count_dict[(first_list[index][1], length_list[index])] += 1
    return pd.DataFrame([[item[0][0], item[0][1], item[1]] for item in count_dict.items()], columns=['first_week',
                                                                                                     'wave_length',
                                                                                                     'count'])


def get_first_last_week_max_length_of_wave(input_df, current_states=['all']):
    """
    This function returns the wave start, wave end, wave length and height of the states specified by the above state
    parameter. A wave start is defined as the first week of a cold weather season in which the threshold of 2.0 infected
    per 100 000 inhabitants is crossed. The outlier year 2009 (swine flu) is excluded.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :param current_states: A list, containing the names of the states relevant for the statistic.
    :return: A dict, containing the first, last week of a wave, the length and the height of a wave in the form of
    four numpy.ndarrays. First, last week and length, height are associated via the index of their numpy.ndarray.
    That is to say same index means that the value belongs to the same wave.
    """

    first_list = []
    last_list = []
    max_list = []

    # Extracting the wave start, end and height from the data frame.
    for state in input_df['state'].unique().tolist():
        if state in current_states or current_states[0] == 'all':
            helper_first_list, helper_last_list, helper_max_list = get_first_last_max_per_state(input_df, state)
            first_list.extend(helper_first_list)
            last_list.extend(helper_last_list)
            max_list.extend(helper_max_list)

    # Calculating the wave lengths.
    length_list = []
    for index in range(len(first_list)):
        if last_list[index][1] < first_list[index][1]:
            length_list.append(last_list[index][1] + 53 - first_list[index][1])
        else:
            length_list.append(last_list[index][1] - first_list[index][1] + 1)

    # Creating boolean index implementing the specified selection.
    no_2009_and_not_waveless_indices = []
    for index in range(len(first_list)):
        current_year_week = first_list[index]
        if (current_year_week[0] != 2009 or current_year_week[1] < 25) and 0 < current_year_week[1]:
            no_2009_and_not_waveless_indices.append(True)
        else:
            no_2009_and_not_waveless_indices.append(False)

    return {'first': list(np.array(first_list)[no_2009_and_not_waveless_indices]),
            'last': list(np.array(last_list)[no_2009_and_not_waveless_indices]),
            'max': list(np.array(max_list)[no_2009_and_not_waveless_indices]),
            'length': list(np.array(length_list)[no_2009_and_not_waveless_indices])}


def get_first_last_max_per_state(input_df, state_str, start_year=2005, end_year=2015, start_week=25, end_week=24,
                                 target_column_name='influenza_week-1', threshold=2.0):
    """
    This function returns the wave start, wave end, wave max of the states specified by the above state
    parameter. The underlying quantity does not have to be the number of influenza cases on a state level and it is
    specified by target column parameter. In the usual case of referring to influenza infections a wave start is defined
    as the first week of a cold weather season in which the threshold of infected per 100 000 inhabitants is crossed.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', target_column_name.
    :param state_str: A str, the name of the state of interest.
    :param start_year: An int, the start year of the interval in which the start, end and max of the waves are
    calculated.
    :param end_year: An int, the end year of the interval in which the start, end and max of the waves are
    calculated.
    :param start_week: An int, the start week of the interval in which the start, end and max of the waves are
    calculated.
    :param end_week: An int, the end week of the interval in which the start, end and max of the waves are
    calculated.
    :param target_column_name: The target column specifying the quantity of interest.
    :param threshold: A float, the threshold specifying the start and the end of a wave.
    :return: A 3-tuple, containing the first, last week of a wave and the max of a wave in the form of three lists.
    First, last week and max are associated via the list index. That is to say same index means that the value belongs
    to the same wave.
    """

    if not ('year_week' in list(input_df.columns) and 'state' in list(input_df.columns)
            and target_column_name in list(input_df.columns)):
        raise ValueError('The input data frame has to have the column names "year_week" and ' + target_column_name
                         + '.')

    first_list = []
    last_list = []
    max_list = []

    state_df = input_df[input_df['state'] == state_str]
    for year in range(start_year, end_year):
        dummy_df, current_year_df = DataFormatting.get_wave_complement_interval_split(state_df, start_year=year, start_week=start_week,
                                                                       end_year=year+1, end_week=end_week)

        max_list.append(current_year_df[target_column_name].max())

        influenza_list = current_year_df[target_column_name].tolist()
        first_index = get_first_or_last_greater_than(influenza_list, threshold)
        last_index = get_first_or_last_greater_than(influenza_list, threshold, first=False)

        if first_index is not None:
            helper_first = current_year_df['year_week'].iloc[first_index]
            helper_last = current_year_df['year_week'].iloc[last_index]

            first_list.append((helper_first[0], helper_first[1] - 1))
            last_list.append((helper_last[0], helper_last[1] - 1))
        else:
            first_list.append((year, -1))
            last_list.append((year, -1))

    return first_list, last_list, max_list


def get_first_or_last_greater_than(input_list, threshold_float=2.0, first=True):
    """
    This function takes an list as input and returns the first (or last if first == False) index for which the
    associated list element is greater than the threshold.

    :param input_list: A list, of floats.
    :param threshold_float: A float, the threshold.
    :param first: A bool, specifying whether the index of the first or last element which crosses the threshold is
    returned.
    :return: An int or None, the index of the first (or last in case first == False) list element which crosses the
    threshold. If all elements are smaller or equal to the threshold None is returned.
    """
    my_range = range(len(input_list))

    if not first:
        my_range = reversed(my_range)

    for index in my_range:
        if threshold_float < input_list[index]:
            return index

    return None


def get_number_of_reported_infected_per_state(input_df):
    """
    This function extracts the overall number of reported influenza infections for each state in Germany and returns
    the respective dictionary. The period from the 25th week of 2009 till the 24th week of 2010 is excluded. In this
    period the "outlier wave" (swine flu) occurred.

    :param input_df: A pandas.DataFrame, containing a rows with names 'state', 'influenza_week-1'.
    :return: A dict, holding the sixteen states of Germany as key. The values are the overall sum of reported influenza
    cases normalized by 100 000 inhabitants.
    """

    start_year_excluded = 2009
    start_week_excluded = 25
    end_year_excluded = 2010
    end_week_excluded = 24

    interval_bool_series = input_df['year_week'].apply(
        lambda x: not((start_year_excluded <= x[0] and (start_week_excluded <= x[1] or start_year_excluded < x[0])) and (
                x[0] <= end_year_excluded and (x[1] <= end_week_excluded or x[0] < end_year_excluded))))

    return sorted(list(input_df[interval_bool_series].groupby('state')['influenza_week-1'].sum().to_dict().items()),
                  key=lambda x: x[1])


###########################
# I.2.) Visualization
###########################

def visualize_overall_reported_cases(input_df):
    """
    This function visualizes the cumulative number of reported influenza infections for each of the sixteen
    states of Germany from 2005 till 2015. The period from the 25th week of 2009 till the 24th week of 2010 is excluded.
    In this period the "outlier wave" (swine flu) occurred.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :return: A bokeh figure, visualizing the sum of the overall reported influenza infections for each of the sixteen
    states of Germany.
    """
    input_list = get_number_of_reported_infected_per_state(input_df)

    states = [state_sum[0] for state_sum in input_list]
    sum_of_rep_cases = [state_sum[1] for state_sum in input_list]

    source = ColumnDataSource(data=dict(states=states, sum_of_rep_cases=sum_of_rep_cases))

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(plot_height=500, plot_width=500, x_range=states, y_range=(0, 900), tools=TOOLS,
               title="Total Number of Reported Influenza Infections from 2005-2015",
               y_axis_label='# Reported Influenza Infections per 100 000 inhabitants')

    p.vbar(x='states', top='sum_of_rep_cases', width=0.9, source=source, legend=False,
           line_color='white', fill_color=linear_cmap(field_name='sum_of_rep_cases', palette=["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"], low=0,
                                                      high=900))
    p.line(x=[0, 16], y=[500, 500], color='black')
    p.line(x=[0, 16], y=[300, 300], color='black')

    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 3.0 / 4
    p.ygrid.grid_line_color = None

    return p


def visualize_infection_numbers_on_map(input_df):
    """
    This function visualized the reported number of influenza infections per state on a map.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :return: A 3 tuple, the first entry is the figure. The second entry are the bokeh patches (the states) and the third
    entry is a year_week_list associated to the currently displayed snapshot of the influenza situation in Germany.
    """

    # Loading the shape file containing the state boundaries.
    dat = shapefile.Reader('Data/GeoData/vg2500_bld.shp')
    # Sorting the array lexicographically by name.
    ele_sorted_by_name_list = sorted([ele for ele in dat.iterRecords()],
                                     key=lambda x: x[3] if type(x[3]) == str else x[3].decode(encoding='windows-1252'))

    states = [ele[2] for ele in ele_sorted_by_name_list]
    x_states_list = []
    y_states_list = []

    for state_name in states:
        data = get_dict(state_name, dat)
        x_states_list.append(data[state_name]['lat_list'][0])
        y_states_list.append(data[state_name]['lng_list'][0])

    TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"

    column_value_for_all_states_per_week_dict, year_week_list = get_column_dict_and_year_week_list(input_df)

    data_dict = dict(x=x_states_list, y=y_states_list, name=input_df['state'].unique().tolist(),
                     rate=column_value_for_all_states_per_week_dict['0'], **column_value_for_all_states_per_week_dict)

    source = ColumnDataSource(data=data_dict)

    custom_colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    color_mapper = LogColorMapper(palette=custom_colors, low=0, high=60)

    p = figure(title="Reported Influenza Infections per State",  tools=TOOLS, h_symmetry=False, v_symmetry=False,
               min_border=0, x_axis_location=None, y_axis_location=None)

    color_bar = ColorBar(color_mapper=color_mapper, major_label_text_font_size="10pt",
                         ticker=FixedTicker(ticks=[2.0, 5.0, 10.0, 20.0, 40.0, 60.0]),
                         label_standoff=7, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [("State", "@name"), ("# of Reported Infections", "@rate")]

    p.grid.grid_line_color = None
    p_patches = p.patches('x', 'y', source=source, fill_color={'field': 'rate', 'transform': color_mapper},
                          fill_alpha=0.9, line_color="black", line_width=0.3)
    return p, p_patches, year_week_list


def get_column_dict_and_year_week_list(input_df, column_name='influenza_week-1'):
    """
    This function provides the repoerted influenza infections for the sixteen states per year, week combination.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :param column_name: A str, specifying the column with respect to which
    :return: A 2-tuple, consisting of a dict and a list. The dict contains for year year, week combination the reported
    influenza numbers for the sixteen states. The list contains the associated year, week combinations.
    """
    # Getting the influenza numbers from all rows except the initial year and week of the time horizon. The reason is
    # that the influenza numbers of this year, week row are not contained in the time horizon.
    year_week_list = input_df['year_week'].unique().tolist()[1:]
    column_dict = {}
    for index, year_week in enumerate(year_week_list):
        column_dict[str(index)] = input_df[input_df['year_week'] == year_week][column_name].tolist()

    # Returning all weeks but the last. The reason is that the influenza numbers for the last week are not contained in
    # this data frame. Since the column with name 'influenza_week-1' refers to the influenza numbers of the last week.
    return column_dict, year_week_list[:-1]


def get_parts(shape_obj):
    """
    This function is from http://www.abisen.com/blog/bokeh-maps/.
    Given a shape_object this function returns a list of lists for latitude and longitudes values.
    This function handles scenarios where there are multiple parts to a shapeObj

    :param shape_obj: A shape object.
    :return: A list of lists for latitude and longitudes values.
    """
    points = []
    num_parts = len(shape_obj.parts)
    end = len(shape_obj.points) - 1
    segments = list(shape_obj.parts) + [end]

    for i in range(num_parts):
        points.append(shape_obj.points[segments[i]:segments[i + 1]])

    return points


def get_dict(state_name, shapefile):
    """
    This function is from http://www.abisen.com/blog/bokeh-maps/.

    :param state_name: A str, the name of the considered state.
    :param shapefile: A shapefile, containing the boundaries of the states.
    :return: A dict, containing the lists of the latitude, longitude coordinates of the considered state and
    the total area.
    """
    state_dict = {state_name: {}}

    rec = []
    shp = []
    points = []

    # Select only the records representing the
    # "state_name" and discard all other
    for i in shapefile.shapeRecords():

        if i.record[2] == state_name:
            rec.append(i.record)
            shp.append(i.shape)

    # In a multi record state for calculating total area
    # sum up the area of all the individual records
    #        - first record element represents area in cms^2
    total_area = sum([float(i[0]) for i in rec]) / (1000*1000)

    # For each selected shape object get
    # list of points while considering the cases where there may be
    # multiple parts  in a single record
    for j in shp:
        for i in get_parts(j):
            points.append(i)

    # Prepare the dictionary
    # Seperate the points into two separate lists of lists (easier for bokeh to consume)
    #      - one representing latitudes
    #      - second representing longitudes
    lat = []
    lng = []
    for i in points:
      lat.append([j[0] for j in i])
      lng.append([j[1] for j in i])

    state_dict[state_name]['lat_list'] = lat
    state_dict[state_name]['lng_list'] = lng
    state_dict[state_name]['total_area'] = total_area

    return state_dict


def visualize_state_commonalities(data_df):
    """
    This function returns a bokeh visualization of the influenza waves for the states of Germany from 2005 till 2015.

    :param data_df: A pandas.DataFrame, containing the influenza progression of the sixteen states of Germany.
    :return: A bokeh figure, visualizing the influenza progressions of the sixteen states of Germany.
    """

    kwargs_state_influenza_dict = OrderedDict()

    for current_state in data_df['state'].unique():

            # State information
            state_indices = data_df['state'] == current_state
            state_df = data_df[state_indices]

            kwargs_state_influenza_dict[current_state] = state_df['influenza_week-1'].tolist()[:-1]

    return plot_results('States of Germany', 'Date', '# Influenza Infections per 100 000 Inhabitants',
                        data_df['year_week'].unique().tolist()[1:], **kwargs_state_influenza_dict)


def plot_results(title_param, x_axis_title_param, y_axis_title_param, dates_list=None, **kwargs):
    """
    A generic function for plotting step functions.

    :param title_param: A str, the title of the plot.
    :param x_axis_title_param:  A str, the x axis title.
    :param y_axis_title_param: A str, the y axis title.
    :param dates_list: A list of dates, used to scale and format the x axis accordingly.
    :param kwargs: A list of float or int, representing the y coordinates.
    :return: A bokeh figure, of a step function plot.
    """

    if kwargs is None:
        raise Exception('kwargs is not allowed to be None.')

    p = figure(plot_width=800, plot_height=500, title=title_param, x_axis_label=x_axis_title_param,
               y_axis_label=y_axis_title_param, toolbar_location="right")

    # In case a dates list is provided as parameter:
    # This list is formatted and then used in the plot.
    if dates_list:
        plot_x = [Week(year_week[0], year_week[1]).monday() for year_week in dates_list]
        p.xaxis.formatter = DatetimeTickFormatter(years=["%D %B %Y"])
        p.xaxis.major_label_orientation = 3.0 / 4
        p.xaxis[0].ticker.desired_num_ticks = 30

    # Coloring different graphs
    color_list = Category20b[16]
    number_of_colors = len(color_list)
    color_index = 0

    for key, argument in kwargs.items():
        if not dates_list:
            plot_x = range(len(argument))

        color_index += 1
        p.step(plot_x, argument, legend=key, color=color_list[color_index % number_of_colors], alpha=1.0,
               muted_color=color_list[color_index % number_of_colors], muted_alpha=0.0)

    p.legend.location = "top_left"
    p.legend.click_policy = "mute"
    p.legend.background_fill_alpha = 0.0

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    return p


def visualize_data_per_state(input_df, state_str="Baden-Wuerttemberg"):
    """
    This function returns a bokeh figure visualizing the influenza, trend, temperature features for the specified state.

    :param input_df: A pandas.DataFrame, containing rows with names 'state', 'year_week', influenza_week-1,
    influenza_germany_week-1, trend_week-1, trend_germany_week-1, temp_mean-1.
    :param state_str: A str, specifying the state.
    :return: A bokeh figure, visualizing the features for the specified state.
    """

    # Only the rows associated to the current state are considered.
    state_indices = input_df['state'] == state_str
    state_df = input_df[state_indices]

    influenza_list = state_df['influenza_week-1'].tolist()[:-1]
    influenza_germany_list = state_df['influenza_germany_week-1'].tolist()[:-1]
    google_trends_list = state_df['trend_week-1'].tolist()[:-1]
    google_trends_germany_list = state_df['trend_germany_week-1'][:-1]
    temp_list = state_df['temp_mean-1'].div(10).tolist()[:-1]
    # humid_list = state_df['humid_mean-1'].tolist()[:-1]
    # prec_list = state_df['prec_mean-1'].tolist()[:-1]
    dates_list = state_df['year_week'].unique().tolist()[1:]

    p = figure(plot_width=800, plot_height=500, title=state_str, x_axis_label='Date')

    plot_x = [Week(year_week[0], year_week[1]).monday() for year_week in dates_list]

    p.xaxis.formatter = DatetimeTickFormatter(
        years=["%D %B %Y"]
    )

    p.xaxis.major_label_orientation = 3.0 / 4
    p.xaxis[0].ticker.desired_num_ticks = 30

    # Influenza Numbers for Current State and for Germany as a Whole
    p.yaxis.axis_label = '# Influenza Infections'
    p.y_range = Range1d(start=0, end=max(max(influenza_list), max(influenza_germany_list))+3)

    # Google Trends Data
    p.extra_y_ranges['trends'] = Range1d(start=0, end=max(max(google_trends_list), max(google_trends_germany_list)))
    p.add_layout(LinearAxis(y_range_name='trends', axis_label='Google Trends Score'), 'left')

    # Temperature
    p.extra_y_ranges['temp'] = Range1d(start=min(temp_list)-1, end=max(temp_list)+2)
    p.add_layout(LinearAxis(y_range_name='temp', axis_label='Temperature in Degrees Celsius'), 'right')

    # # Precipitation
    # p.extra_y_ranges['prec'] = Range1d(start=0, end=500)
    # p.add_layout(LinearAxis(y_range_name='prec', axis_label='Prec in'), 'right')

    keys_list = ['Influenza', 'Influenza Germany', 'Google Trends', 'Google Trends Germany', 'Temperature']
    argument_list = [influenza_list, influenza_germany_list, google_trends_list, google_trends_germany_list, temp_list]
    color_list = ['#000000', '#5b5b5b', '#1a6864', '#2a9e98', '#c11515']
    y_range_list = ['dummy', 'dummy', 'trends', 'trends', 'temp']
    alpha_list = [1.0, 1.0, 1.0, 1.0, 0.0]
    muted_alpha_list = [0.0, 0.0, 0.0, 0.0, 1.0]

    for index in range(0, 2):
        p.step(plot_x, argument_list[index], legend=keys_list[index], color=color_list[index], alpha=alpha_list[index],
               muted_color=color_list[index], muted_alpha=muted_alpha_list[index])

    for index in range(2, 5):
        p.step(plot_x, argument_list[index], legend=keys_list[index], color=color_list[index], alpha=alpha_list[index],
               muted_color=color_list[index], muted_alpha=muted_alpha_list[index], y_range_name=y_range_list[index])

    p.legend.location = "top_left"
    p.legend.click_policy = "mute"
    p.legend.background_fill_alpha = 0.0

    return p


def visualize_wave_stats_distributions(input_df, states=['all']):
    """
    This function provides statistics about the influenza wave on a state level in the form of a figure.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :param states: A list, containing the names of the states relevant for the statistic.
    :return: A Bokeh Column, containing the four figures visualizing wave statistics.
    """

    # Getting the wave statistics.
    first_last_week_max_length_of_wave_dict = get_first_last_week_max_length_of_wave(input_df, current_states=states)

    first_list = [year_week[1] for year_week in first_last_week_max_length_of_wave_dict['first']]
    last_list = [year_week[1] for year_week in first_last_week_max_length_of_wave_dict['last']]
    max_list = first_last_week_max_length_of_wave_dict['max']
    length_list = first_last_week_max_length_of_wave_dict['length']

    value_list_list = [first_list, last_list, max_list, length_list]
    title_list = ['Week of Year the Wave Started', 'Week of Year the Wave Ended',
                  'Severity of the Wave in Infected per 100 000 Inhabitants', 'Duration of the Wave in Weeks']
    x_labels_list = ['Week of the Year', 'Week of the Year', 'Infected per 100 000 Inhabitants', 'Duration in Weeks']
    p_list = []

    # Generating the histogram and estimated density.
    for index in range(4):
        min_value = min(value_list_list[index])
        max_value = max(value_list_list[index])
        num_of_bins = int(max_value-min_value)

        hist, edges = np.histogram(value_list_list[index], density=True,
                                   bins=num_of_bins)

        p = figure(title=title_list[index], x_axis_label=x_labels_list[index], y_axis_label='Probabilities',
                   plot_width=400, plot_height=350,)

        p.xaxis[0].ticker.desired_num_ticks = 20

        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="#036564", line_color="#033649")

        x_gird = np.linspace(start=min_value-1, stop=max_value+1, num=2*num_of_bins)
        p.line(np.array(x_gird), kde_sklearn(np.array(value_list_list[index]), np.array(x_gird), bandwidth=1.5), color='black')
        p_list.append(p)

    row1 = row(p_list[0], p_list[1])
    row2 = row(p_list[2], p_list[3])

    return column(row1, row2)


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """
    This function returns the estimated density associated to the samples in x. Kernel density estimation is used to
    estimate the density.

    :param x: A numpy.ndarray, containing the samples.
    :param x_grid: A numpy.ndarray, containing the x coordinates at which the estimated density is evaluated.
    :param bandwidth: A float, the bandwidth for the density estimation.
    :param kwargs:
    :return: A numpy.ndarray, the values of the estimated density when evaluated at the grid points.
    """
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)

    kde_skl.fit(x[:, np.newaxis])

    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])

    return np.exp(log_pdf)


def visualize_wave_start_vs_severity_via_box(input_df, states=['all']):
    """
    The relation between the wave start and the severity of the wave is visualized via a box plot.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :param states: A list, containing the relevant state names.
    :return: A bokeh figure, visualizing the relation between wave start week and severity via a box plot.
    """

    first_last_max_length_lists = list(get_first_last_week_max_length_of_wave(input_df, current_states=states).items())
    first_list = first_last_max_length_lists[0][1]

    week_list = ["Week " + str(year_week[1]) for year_week in first_list]
    max_value_of_wave_list = first_last_max_length_lists[2][1]

    week_categories_unique_list = sorted(list(set(week_list)))
    week_categories_display_order_list = sorted(week_categories_unique_list, key=lambda x: int(x[-2:].strip()))

    return box_plot(week_list, max_value_of_wave_list, week_categories_display_order_list,
                    title="Wave Start vs Wave Severity", x_axis_label="Calender Week",
                    y_axis_label='Number of Infected per 100 000 Inhabitants')


def visualize_wave_start_vs_length_via_box(input_df, states=['all']):
    """
    The relation between the wave start and the length of the wave is visualized via a box plot.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :param states: A list, containing the relevant state names.
    :return: A bokeh figure, visualizing the relation between wave start week and length via a box plot.
    """

    first_last_max_length_lists = list(get_first_last_week_max_length_of_wave(input_df, current_states=states).items())

    start_length_count_df = get_first_length_count_df(first_last_max_length_lists[0][1],
                                                      first_last_max_length_lists[3][1])

    x_count_list = ["Week " + str(week) for week in start_length_count_df['first_week'].tolist()]
    y_count_list = start_length_count_df['wave_length'].tolist()
    count_for_size_list = start_length_count_df['count'].tolist()

    first_week_list = ["Week " + str(year_week[1]) for year_week in first_last_max_length_lists[0][1]]
    wave_length_list = first_last_max_length_lists[3][1]

    week_categories_unique_list = sorted(list(set(first_week_list)))
    week_categories_display_order_list = sorted(week_categories_unique_list, key=lambda x: int(x[-2:].strip()))

    box_fig = box_plot(first_week_list, wave_length_list, week_categories_display_order_list,
                       title="Wave Start vs Wave Length in Weeks", x_axis_label="Calender Week",
                       y_axis_label="Calender Week")
    # Encoding the count of the start week, wave length pair in the x size.
    box_fig.x(x_count_list, y_count_list, color='black', size=np.array(count_for_size_list)*3)

    return box_fig


def box_plot(x_list, y_list, x_cat_unique_display_order_list, title="", x_axis_label="", y_axis_label=""):
    """
    This function returns a box plot. This is a modified version of
    https://bokeh.pydata.org/en/latest/docs/gallery/boxplot.html.

    :param x_list: A list, containing the categories.
    :param y_list: A list, containing the values associated to the categories.
    :param x_cat_unique_display_order_list: A list, containing the categories in a specific order. The categories will
    be displayed on the x-axis in this order.
    :param title: A str, the title of the figure.
    :param x_axis_label: A str, the x axis label.
    :param y_axis_label: A str, the y axis label.
    :return: A bokeh figure, a box plot.
    """

    x_categories_unique_sorted_list = sorted(list(set(x_list)))
    first_max_df = pd.DataFrame(columns=['group', 'score'])
    first_max_df[first_max_df.columns[0]] = x_list
    first_max_df[first_max_df.columns[1]] = y_list

    # Find the quartiles and IQR for each category
    groups = first_max_df.groupby('group')
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    # Find the outliers for each category
    def outliers(group):
        cat = group.name
        return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']

    out = groups.apply(outliers).dropna()

    # Prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = []
        outy = []
        for cat in x_categories_unique_sorted_list:
            # Only add outliers if they exist
            if not out.loc[cat].empty:
                for value in out[cat]:
                    outx.append(cat)
                    outy.append(value)

    p = figure(title=title, plot_width=400, plot_height=350, x_range=x_cat_unique_display_order_list,
               x_axis_label=x_axis_label, y_axis_label=y_axis_label)
    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.score = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, 'score']), upper.score)]
    lower.score = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, 'score']), lower.score)]

    # stems
    p.segment(x_categories_unique_sorted_list, upper.score, x_categories_unique_sorted_list, q3.score, line_color="black")
    p.segment(x_categories_unique_sorted_list, lower.score, x_categories_unique_sorted_list, q1.score, line_color="black")

    # boxes
    p.vbar(x_categories_unique_sorted_list, 0.7, q2.score, q3.score, fill_color="#E08E79", line_color="black")
    p.vbar(x_categories_unique_sorted_list, 0.7, q1.score, q2.score, fill_color="#3B8686", line_color="black")

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(x_categories_unique_sorted_list, lower.score, 0.2, 0.01, line_color="black")
    p.rect(x_categories_unique_sorted_list, upper.score, 0.2, 0.01, line_color="black")

    # outliers
    if not out.empty:
        p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

    p.x(x_list, y_list, color='black')

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.grid.grid_line_width = 2
    p.xaxis.major_label_orientation = 3.0 / 4

    return p


def visualize_wave_start_vs_severity_via_violin(input_df, states=['all']):
    """
    This function provides a violin plot of the wave severity given the start week of the wave.

    :param input_df: A pandas.DataFrame, containing rows with name 'year_week', 'state' and 'influenza_week-1'.
    :param states: A list[str], of names of the considered state.
    :return: A bokeh figure, a violin plot of the severity of the wave or in other words the highest peak per wave
    start week.
    """
    first_last_max_length_lists = list(get_first_last_week_max_length_of_wave(input_df, current_states=states).items())
    first_list = first_last_max_length_lists[0][1]
    max_list = first_last_max_length_lists[2][1]

    first_max_df = pd.DataFrame(columns=['Week of Year Wave Started',
                                         'Wave Peak in Number of Infected per 100 000 Inhabitants'])
    first_max_df[first_max_df.columns[0]] = [year_week[1] for year_week in first_list]
    first_max_df[first_max_df.columns[1]] = max_list

    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")
    plot_dims = (15.0, 8.27)
    plt.subplots(figsize=plot_dims)
    ax = sns.violinplot(x=first_max_df.columns[0], y=first_max_df.columns[1], data=first_max_df,
                        palette="Set3", split=False,
                        scale="count", inner="stick", label='big')
    ax.set_title("Wave Start vs Wave Severity")
    return plt


def visualize_wave_start_vs_length_via_heatmap(input_df, states=['all']):
    """
    This function provides a heatmap of the start week of the wave versus the wave length.

    :param input_df: A pandas.DataFrame, containing rows with name 'year_week', 'state' and 'influenza_week-1'.
    :param states: A list[str], of names of the considered state.
    :return: A bokeh figure, a heatmap of the start week of the wave versus the wave length.
    """
    first_last_max_length_lists = list(get_first_last_week_max_length_of_wave(input_df, current_states=states).items())
    start_length_count_df = get_first_length_count_df(first_last_max_length_lists[0][1],
                                                      first_last_max_length_lists[3][1])

    # Categories for the x and y-axis.
    wave_start_list = [str(i) for i in range(1, 18)]
    wave_length_list = [str(i) for i in range(1, 18)]

    # Coloring
    colors = ["#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    color_map_min = start_length_count_df['count'].min()
    color_map_max = start_length_count_df['count'].max()
    if color_map_min == color_map_max:
        color_map_max += 1
    mapper = LinearColorMapper(palette=colors, low=1, high=max(color_map_max, 5))

    source = ColumnDataSource(start_length_count_df)

    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    title_str = 'Wave Start vs Wave Length in Weeks'

    p = figure(title=title_str,
               x_range=wave_start_list, y_range=list(wave_length_list),
               plot_width=900, plot_height=400,
               tools=TOOLS, toolbar_location='above',
               x_axis_label='Week of Year the Wave Started',
               y_axis_label='Wave Length in Weeks')

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "7pt"
    p.axis.major_label_standoff = 0

    p.rect(x="first_week", y="wave_length", width=1, height=1,
           source=source,
           fill_color={'field': 'count', 'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    p.select_one(HoverTool).tooltips = [
        ('Wave Start, Wave Length', '@first_week, @wave_length'),
        ('Count', '@count'),
    ]

    return p


#######################
#######################
# II.) Classification
#######################
#######################

class ForecastProvider(object):

    def __init__(self, data_frame_provider=DataFrameProvider(), states_list=['all'], no_2009_bool=True,
                 only_seasonal_weeks_bool=False, weeks_in_advance_int=11, classification_pipeline_per_week_list=
                 [Pipeline(steps=[('preprocessing', StandardScaler()),
                            ('regressor', MLPClassifier(hidden_layer_sizes=(20, 20, 10), alpha=1,
                                                        learning_rate='adaptive', batch_size=3000,
                                                        random_state=1341, max_iter=1000))])]*11,
                 excluded_features_list=['prec_', 'humid_', 'temp_', '12', '11']):
        """
        This class provides methods which either perform a classification for either a specific forecasting period or
        the complete classification horizon specified by the weeks_in_advance_int parameter. The method
        do_cross_validation can be used for model and feature selection initializing different instances of this class
        for different models and feature sets and calling the method. The do_gridsearch method in turn can be used to
        select the ideal parameters for a model. For a more detailed analysis for specific years. the
        classificatoin_forecast_for_year method can be used. The get_formatted_pred_probas_and_actual_values provides
        a comprehensive evaluation with respect to all years. In addition, get_formatted_pred_probas_and_actual_values
        returns the data needed for the animation presented via the associated webapp.

        :param data_frame_provider: A AbstractDataFrameProvider, providing the data for the forecast.
        :param states_list: A list[str], of state names which are not removed from the data set.
        :param no_2009_bool: A bool, indicating whether the outlier year 2009 in which the swine flu occurred should be
        excluded.
        :param only_seasonal_weeks_bool: A bool, indicating whether only data from the cold weather season of a year should
        be included.
        :param weeks_in_advance_int: An int, the number of forecasting weeks.
        :param classification_pipeline_per_week_list: A list, of classification pipelines or models each entry
        corresponds to one forecasting distance.
        :param excluded_features_list: A list[str], containing strs which lead to excluding a column with a name containing
        str.
        """

        if weeks_in_advance_int != len(classification_pipeline_per_week_list):
            raise ValueError(
                'The length of the list of classification pipelines has to be equal to the weeks in advance '
                'parameter.')

        self.states_list = states_list
        self.no_2009_bool = no_2009_bool
        self.only_seasonal_weeks_bool = only_seasonal_weeks_bool
        self.weeks_in_advance_int = weeks_in_advance_int
        self.classification_pipeline_per_week_list = classification_pipeline_per_week_list
        self.excluded_features_list = excluded_features_list
        # This class provides the data.
        self.data_frame_provider = data_frame_provider

        # Generating Features and Target Variables

        # Getting the feature data frame and selecting the time steps over which the min, max, mean and variance of the
        # weather is calculated.
        weather_trend_influenza_df = data_frame_provider.getFeaturesDF()
        # Storing the list of year, week tuples.
        self.complete_unique_year_week_list = weather_trend_influenza_df['year_week'].unique()

        # Getting the target variable associated with the features.
        y_df_series = data_frame_provider.rawYDataFrame

        # Excluding specific states, years, periods of the year. Removing rows.
        weather_trend_influenza_df, y_df_series = DataFormatting.exclude_rows_by_states_2009_summer(
            weather_trend_influenza_df, y_df_series, valid_states_list=states_list,
            no_2009_bool=no_2009_bool, only_seasonal_weeks_bool=only_seasonal_weeks_bool)

        # Shifting the target variable. Such that a 2,3, ... weeks forecast performed. In addition it is possible add
        # additional columns to the target variable. Thus the target variable could consist for example of the next week and
        # the week afterwards.
        weather_trend_influenza_df, self.y_df_series = DataFormatting.shift_append_target_weeks(
            weather_trend_influenza_df, y_df_series, shift=0, number_of_additional_weeks=weeks_in_advance_int - 1)

        # Dropping features. Removing columns.
        self.weather_trend_influenza_df = DataFormatting.remove_columns_with(weather_trend_influenza_df,
                                                                        excluded_features_list)

        # Will be used to select the state specific influenza columns.
        self.number_influenza_state_columns = sum(
            [re.search(r'influenza_week', ele) is not None for ele in self.weather_trend_influenza_df.columns])

    def do_gridsearch(self, classification_pipeline=Pipeline(
                     steps=[('preprocessing', StandardScaler()),
                            ('regressor', MLPClassifier(hidden_layer_sizes=(20, 20, 10), alpha=1,
                                                        learning_rate='adaptive', batch_size=3000,
                                                        random_state=1341, max_iter=1000))]),
                      parameters={'regressor__batch_size': [500, 3000, 7000],
                                  'regressor__hidden_layer_sizes': [(10, 10), (10, 20, 10)],
                                  'regressor__alpha': [0.001, 0.1, 1.0]},
                      forecasting_week_index=0, wave_threshold=0.8):
        """
        This method implements a grid search for the specified classification pipeline and pipline parameters based
        on the instance variables.
        :param classification_pipeline: A sklearn.pipeline.Pipeline or another sklearn classifier, used for the
        classification task.
        :param parameters: A dict, containing the pipeline and classifier parameters for the grid search.
        :param forecasting_week_index: An int, the index of the forecasting week, this has to be smaller or equal to
        weeks_in_advance_int - 1.
        :param wave_threshold: A float, the threshold for the number of reported infections. If the y values
        (number of reported influenza infections) crosses this threshold the y value is classified as 1 otherwise as 0.
        :return: A tuple, containing the best scores and the best parameters.
        """

        weather_trend_influenza_df = self.weather_trend_influenza_df.copy(deep=True)
        y_df_series = self.y_df_series.copy(deep=True)

        if isinstance(y_df_series, pd.DataFrame):
            y_ndarray = y_df_series.as_matrix().astype(np.float64)
        else:
            y_ndarray = y_df_series.values.reshape(-1, 1).astype(np.float64)

        #     #
        #     # Uncomment the following line if not sure about how to refer to the parameters.
        #     #
        #     print('Available Params:')
        #     print(regressor.get_params().keys())

        # Getting the cross validation split indices.
        cv_indices_list = DataFormatting.get_custom_cv_split_index_list(weather_trend_influenza_df, start_week=25,
                                                                        end_week=24, year_range_start=2005,
                                                                        year_range_end=2014,
                                                                        exclude_2009_bool=self.no_2009_bool)

        print('Grid Search:')
        t_start = datetime.now()
        grid_search = GridSearchCV(classification_pipeline, parameters, cv=cv_indices_list)
        grid_search.fit(weather_trend_influenza_df[weather_trend_influenza_df.columns[2:]].as_matrix().
                        astype(np.float64), (wave_threshold <= y_ndarray[:, forecasting_week_index]).astype(int))
        print('elapsed time: %0.3fs' % (datetime.now() - t_start).total_seconds())

        print('Best Score:')
        print(grid_search.best_score_)
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print(param_name)
            print(best_parameters[param_name])

        return grid_search.best_score_, best_parameters

    def do_cross_validation(self, classification_pipeline=Pipeline(
                     steps=[('preprocessing', StandardScaler()),
                            ('regressor', MLPClassifier(hidden_layer_sizes=(20, 20, 10), alpha=1,
                                                        learning_rate='adaptive', batch_size=3000,
                                                        random_state=1341, max_iter=1000))]), forecasting_week_index=0,
                            wave_threshold=0.8):
        """
        This method performs a cross-valdiation. The different validation sets correspond to different validation years.
        The train and test scores for the accuracy, precision, recall, neg_log_loss and roc_auc are printed and
        returned and printed.
        :param classification_pipeline: A sklearn.pipeline.Pipeline or another sklearn classifier, used for the
        classification task.
        :param forecasting_week_index: An int, the index of the forecasting week, this has to be smaller or equal to
        weeks_in_advance_int - 1.
        :param wave_threshold: A float, the threshold for the number of reported infections. If the y values
        (number of reported influenza infections) crosses this threshold the y value is classified as 1 otherwise as 0.
        :return: A dict, containing the scores of the model.
        """

        weather_trend_influenza_df = self.weather_trend_influenza_df.copy(deep=True)
        y_df_series = self.y_df_series.copy(deep=True)

        if isinstance(y_df_series, pd.DataFrame):
            y_ndarray = y_df_series.as_matrix().astype(np.float64)
        else:
            y_ndarray = y_df_series.values.reshape(-1, 1).astype(np.float64)

        # The cross validation is performed by the following train and validation split.
        # Nine validations are performed. One for each of the 9 years of the training set.
        # In other words each of the 9 years is once the validation set and the remaining 8
        # years are once the training set.

        # Getting the cross validation split indices.
        cv_indices_list = DataFormatting.get_custom_cv_split_index_list(weather_trend_influenza_df, start_week=25,
                                                                        end_week=24, year_range_start=2005,
                                                                        year_range_end=2014,
                                                                        exclude_2009_bool=self.no_2009_bool)

        scoring = {'accuracy': 'accuracy',
                   'precision': 'precision', 'recall': 'recall', 'neg_log_loss': 'neg_log_loss', 'roc_auc': 'roc_auc'}
        # Performing cross validation for the model and the benchmark
        scores_model = cross_validate(classification_pipeline, weather_trend_influenza_df[weather_trend_influenza_df.
                                      columns[2:]].as_matrix().astype(np.float64),
                                      (wave_threshold <= y_ndarray[:, forecasting_week_index]).astype(int),
                                      cv=cv_indices_list, scoring=scoring, return_train_score=True)

        keys = ['train_accuracy', 'test_accuracy',
                'train_precision', 'test_precision', 'train_recall',
                'test_recall', 'train_neg_log_loss', 'test_neg_log_loss', 'train_roc_auc', 'test_roc_auc']

        # Printing test and train errors and scores for the benchmark and the model.
        print('Scores:')
        for key in keys:
            print('Model ' + key + ':')
            print(scores_model[key])

        return scores_model

    def classification_forecast_for_year(self, prediction_year=2014, wave_threshold=0.8, probability_threshold=0.5):
        """
        Ths method performs a forecast for the specified year. The training is performed on the remaining years.
        For each forecasting week a separate model is trained. The results for each forecasting week are printed.
        After the model training has finished the ROCs are visualized as well as a figure showing the actual number
        of reported influenza cases and the corresponding prediction probabilities per forecasting week.

        :param prediction_year: An int, the prediction (test) year.
        :param wave_threshold: A float, the threshold for the number of reported infections. If the y values
        (number of reported influenza infections) crosses this threshold the y value is classified as 1 otherwise as 0.
        :param probability_threshold: A float, a probability thresholds. The probability threshold indicates from what
        prediction probability onward a 1 is predicted.
        :return: A tuple, containing the features, target vector, the predictions and a list of year,week tuples.
        """

        weather_trend_influenza_df = self.weather_trend_influenza_df.copy(deep=True)
        y_df_series = self.y_df_series.copy(deep=True)

        # Printing the selected features.
        print('The column names of the data frame:')
        print(weather_trend_influenza_df.columns)

        ###
        # Training and Formatting Results
        ###

        # Test / Train Split
        train_data_df, test_data_df = DataFormatting.get_wave_complement_interval_split(weather_trend_influenza_df,
                                                                                        prediction_year, 25,
                                                                                        prediction_year + 1, 24)

        y_train_df_series = y_df_series.iloc[train_data_df.index]
        y_test_df_series = y_df_series.iloc[test_data_df.index]

        test_year_week_per_row_list = test_data_df['year_week'].tolist()

        # Resetting the indices of the DataFrames and series such that there are no gaps in the index
        # (e.g. 1,2,3,4, ... vs 1,4, ... ). Therefore the indices match the row indices of the numpy
        # arrays storing the features and target values.
        train_data_df = train_data_df.reset_index(drop=True)
        test_data_df = test_data_df.reset_index(drop=True)
        y_train_df_series = y_train_df_series.reset_index(drop=True)
        y_test_df_series = y_test_df_series.reset_index(drop=True)

        # Creating numpy arrays and matrices.
        X_train = train_data_df[train_data_df.columns[2:]].as_matrix().astype(np.float64)
        X_test = test_data_df[test_data_df.columns[2:]].as_matrix().astype(np.float64)

        if isinstance(y_train_df_series, pd.DataFrame):
            y_train = y_train_df_series.as_matrix().astype(np.float64)
            y_test = y_test_df_series.as_matrix().astype(np.float64)
        else:
            y_train = y_train_df_series.values.reshape(-1, 1).astype(np.float64)
            y_test = y_test_df_series.values.reshape(-1, 1).astype(np.float64)

        prediction_ndarray = None
        roc_auc_curve_fig_list = []
        true_vs_pred_per_week_fig_list = []

        for week_index in range(self.weeks_in_advance_int):
            print('')
            print('Current Week Index ' + str(week_index) + ':')

            X_train_cur = X_train

            y_train_current_ndarray = y_train[:, week_index].reshape(-1, 1)
            y_test_current_ndarray = y_test[:, week_index].reshape(-1, 1)

            y_train_current_classifcation_ndarray = (wave_threshold <= y_train_current_ndarray).astype(int)
            y_test_current_classification_ndarray = (wave_threshold <= y_test_current_ndarray).astype(int)

            # The data set is imbalanced therefore examples of the rare class (number of influenza cases above the
            # 0.8 is already relatively rare) are simply copied multiple times.

            multiplication_factor_to_counter_imbalance_int = int(
                y_train_current_classifcation_ndarray.shape[0] // sum(
                    y_train_current_classifcation_ndarray) - 1)

            reach_threshold_bool_indices_train_ndarray = (
                    wave_threshold <= y_train_current_ndarray).flatten()

            X_train_threshold_reached = X_train_cur[reach_threshold_bool_indices_train_ndarray, :]
            y_train_classification_threshold_reached = y_train_current_classifcation_ndarray[
                reach_threshold_bool_indices_train_ndarray]

            for _ in range(multiplication_factor_to_counter_imbalance_int):
                X_train_cur = np.vstack([X_train_cur, X_train_threshold_reached])
                y_train_current_classifcation_ndarray = np.vstack(
                    [y_train_current_classifcation_ndarray, y_train_classification_threshold_reached])

            classification_pipeline = self.classification_pipeline_per_week_list[week_index]
            classification_pipeline.fit(X_train_cur, y_train_current_classifcation_ndarray.flatten())

            # Checking if the classifier has the predict_proba() method otherwise predict() is called.
            try:
                y_test_pred_proba_ndar = classification_pipeline.predict_proba(X_test)[:, 1]
                # Checking if class 1 examples are contained in the target variable vector.
                if 0 < sum(y_test_current_classification_ndarray):
                    # Generating ROC curve figure
                    fpr, tpr, thresholds = roc_curve(y_test_current_classification_ndarray,
                                                     y_test_pred_proba_ndar)
                    fprtpr_plot = figure(title="ROC for " + str(week_index + 1) + " Week Forecast", x_axis_label="False Positive Rate",
                                         y_axis_label="True Positive Rate")
                    fprtpr_plot.line(fpr, tpr)
                    roc_auc_curve_fig_list.append(fprtpr_plot)
                    print("ROC AUC")
                    print(roc_auc_score(y_test_current_classification_ndarray, y_test_pred_proba_ndar))
                    print("Log Loss")
                    print(log_loss(y_test_current_classification_ndarray, y_test_pred_proba_ndar))

            except AttributeError:
                y_test_pred_proba_ndar = classification_pipeline.predict(X_test)

            y_test_pred_classes_ndar = (probability_threshold < y_test_pred_proba_ndar).astype(int).reshape(-1, 1)

            print('Accuracy: ' + str(accuracy_score(y_test_current_classification_ndarray, y_test_pred_classes_ndar)))
            print('Precision: ' + str(precision_score(y_test_current_classification_ndarray, y_test_pred_classes_ndar)))
            print('Recall: ' + str(recall_score(y_test_current_classification_ndarray, y_test_pred_classes_ndar)))

            p = figure(title="Actual Influenza Numbers vs. Prediction Probabilities for " + str(week_index + 1) +
                             " Week Forecast", y_axis_label="# Infections / Prediction Probabilities")
            p.line(range(y_test_current_ndarray.shape[0]), y_test_current_ndarray.flatten(), color='black',
                   legend="Acutal Influenza Numbers")
            p.line(range(y_test_pred_proba_ndar.shape[0]), y_test_pred_proba_ndar, color='red',
                   legend="Prediction Probabilities")
            true_vs_pred_per_week_fig_list.append(p)

            if prediction_ndarray is None:
                prediction_ndarray = y_test_pred_classes_ndar
            else:
                prediction_ndarray = np.hstack([prediction_ndarray, y_test_pred_classes_ndar])

        # Showing the Figures
        if 0 < len(roc_auc_curve_fig_list):
            show(column([row(roc_auc_curve_fig_list), row(true_vs_pred_per_week_fig_list)]))
        else:
            show(row(true_vs_pred_per_week_fig_list))

        return X_test[:, -self.number_influenza_state_columns:], y_test, prediction_ndarray,\
               test_year_week_per_row_list, self.complete_unique_year_week_list

    def get_formatted_pred_probas_and_actual_values(self, wave_thresholds=[0.8, 7.0]):
        """
        This function trains for each forecasting distance, threshold and training set a model and returns the predictions
        for the corresponding test (more precisely validation) year as well as the actual values for that period grouped
        by forecasting week and grouped by state and test (more precisely validation) year. Temporal data is also
        returned.

        :param wave_thresholds: A list[float], of two thresholds. If the y values (number of reported influenza infections)
        crosses this threshold the y value is classified as 1 otherwise as 0.
        :return: A 3-tuple, containing one list and two dicts. The list contains the actual values and the predictions
        probabilities grouped by week of forecast then by threshold and then by test(more precisely validation) year. The
        first dictionary groups test (more precisely validation) data and predictions by state and test (more precisely
        valdiation) year. The second dictionary provides temporal data.
        """

        weather_trend_influenza_df = self.weather_trend_influenza_df.copy(deep=True)
        y_df_series = self.y_df_series.copy(deep=True)

        # Printing the selected features.
        print('The column names of the data frame:')
        print(weather_trend_influenza_df.columns)

        # Specifying the time horizon of the model
        year_range = range(2005, 2015)
        if self.no_2009_bool:
            year_range = [year for year in range(2005, 2015) if year != 2009]

        ###
        # Training and Formatting Results
        ###

        # The returned values.
        output_by_week_list = []
        output_by_year_state_dict = {}
        output_temporal_dict = {'complete_year_week_horizon': self.complete_unique_year_week_list,
                                'year_week_row_by_year': []}  # fpslide Test

        for week_index in range(self.weeks_in_advance_int):
            print('')
            print('Week Index: ' + str(week_index))
            print('Training and Formatting Started:')
            cur_week_output_by_thresholds_list = []
            output_by_week_list.append(
                {'week': week_index + 1, 'output_by_threshold': cur_week_output_by_thresholds_list})

            for index_wave_threshold in range(len(wave_thresholds)):
                print('Threshold Index: ' + str(index_wave_threshold))
                cur_week_thresh_output_list = []
                cur_week_output_by_thresholds_list.append({'threshold': wave_thresholds[index_wave_threshold],
                                                           'output_per_year': cur_week_thresh_output_list})

                for prediction_year_int in year_range:
                    print('Year: ' + str(prediction_year_int))

                    # Test / Train Split

                    # Train and test set are not complements but there is a safety margin separating train and test set.
                    # To ensure that the training set does not contain too much information about the test set.
                    train_test_margin = max(self.weeks_in_advance_int, 10)

                    # Not entirely consistent for years with 53 weeks.
                    if 52 < 24 + train_test_margin:
                        end_week_safety_margin = 24 + train_test_margin - 52
                        end_year_safety_margin = prediction_year_int + 2
                    else:
                        end_week_safety_margin = 24 + train_test_margin
                        end_year_safety_margin = prediction_year_int + 1
                    if 25 - train_test_margin < 1:
                        start_week_safety_margin = 52 + (25 - train_test_margin)
                        start_year_safety_margin = prediction_year_int - 1
                    else:
                        start_week_safety_margin = 25 - train_test_margin
                        start_year_safety_margin = prediction_year_int

                    # Feature Training Set:
                    train_data_df, _ = DataFormatting.get_wave_complement_interval_split(weather_trend_influenza_df,
                                                                                         start_year_safety_margin,
                                                                                         start_week_safety_margin,
                                                                                         end_year_safety_margin,
                                                                                         end_week_safety_margin)

                    # Feature Test Set:
                    _, test_data_df = DataFormatting.get_wave_complement_interval_split(weather_trend_influenza_df,
                                                                                        prediction_year_int, 25,
                                                                                        prediction_year_int + 1, 24)

                    # Target Train and Test Set:
                    y_train_df_series = y_df_series.iloc[train_data_df.index]
                    y_test_df_series = y_df_series.iloc[test_data_df.index]

                    # Resetting the indices of the pandas.DataFrames and pandas.Series such that there are no gaps in the
                    # index (e.g. 1,2,3,4, ... vs 1,4, ... ). Therefore the indices match the row indices of the numpy
                    # arrays storing the features and target values.
                    train_data_df = train_data_df.reset_index(drop=True)
                    test_data_df = test_data_df.reset_index(drop=True)
                    y_train_df_series = y_train_df_series.reset_index(drop=True)
                    y_test_df_series = y_test_df_series.reset_index(drop=True)

                    # Creating numpy arrays and matrices.
                    X_train = train_data_df[train_data_df.columns[2:]].as_matrix().astype(np.float64)
                    X_test = test_data_df[test_data_df.columns[2:]].as_matrix().astype(np.float64)

                    # Calling methods according to the type.
                    if isinstance(y_train_df_series, pd.DataFrame):
                        y_train = y_train_df_series.as_matrix().astype(np.float64)
                        y_test = y_test_df_series.as_matrix().astype(np.float64)
                    else:
                        y_train = y_train_df_series.values.reshape(-1, 1).astype(np.float64)
                        y_test = y_test_df_series.values.reshape(-1, 1).astype(np.float64)

                    # Transforming the continuous values of the target variable into 0 or 1 according to classification
                    # threshold.

                    X_train_cur = X_train

                    y_train_current_ndarray = y_train[:, week_index].reshape(-1, 1)
                    y_test_current_ndarray = y_test[:, week_index].reshape(-1, 1)

                    y_train_current_classifcation_ndarray = (
                            wave_thresholds[index_wave_threshold] <= y_train_current_ndarray).astype(int)
                    y_test_current_classification_ndarray = (
                            wave_thresholds[index_wave_threshold] <= y_test_current_ndarray).astype(int)

                    # The data set is imbalanced therefore examples of the rare class (number of influenza cases above the
                    # 0.8 is already relatively rare) are simply copied multiple times.

                    multiplication_factor_to_counter_imbalance_int = int(
                        y_train_current_classifcation_ndarray.shape[0] // sum(
                            y_train_current_classifcation_ndarray) - 1)

                    reach_threshold_bool_indices_train_ndarray = (
                            wave_thresholds[index_wave_threshold] <= y_train_current_ndarray).flatten()

                    X_train_threshold_reached = X_train_cur[reach_threshold_bool_indices_train_ndarray, :]
                    y_train_classification_threshold_reached = y_train_current_classifcation_ndarray[
                        reach_threshold_bool_indices_train_ndarray]

                    for _ in range(multiplication_factor_to_counter_imbalance_int):
                        X_train_cur = np.vstack([X_train_cur, X_train_threshold_reached])
                        y_train_current_classifcation_ndarray = np.vstack(
                            [y_train_current_classifcation_ndarray, y_train_classification_threshold_reached])

                    # The classification model for the current forecasting distance is assigned.
                    classification_pipeline = self.classification_pipeline_per_week_list[week_index]

                    classification_pipeline.fit(X_train_cur, y_train_current_classifcation_ndarray.flatten())

                    # Constructing 2 of the return objects (output_by_year_state_dict, output_temporal_dict).

                    # Constructing per year temporal, X_test and y_test data.
                    if week_index == 0 and index_wave_threshold == 0:
                        output_temporal_dict['year_week_row_by_year'].append(test_data_df['year_week'].unique())

                    # Loop over all states
                    for state in test_data_df['state'].unique():
                        index_cur_state = test_data_df['state'] == state

                        # Checking if the classifier has the predict_proba() method otherwise predict() is called.
                        try:
                            cur_state_y_test_pred_proba_ndar = classification_pipeline.predict_proba(
                                X_test[index_cur_state, :])[:, 1]
                        except AttributeError:
                            cur_state_y_test_pred_proba_ndar = classification_pipeline.predict(
                                X_test[index_cur_state, :])

                        current_prediction = cur_state_y_test_pred_proba_ndar.reshape(-1, 1)

                        if week_index == 0 and index_wave_threshold == 0:
                            output_by_year_state_dict[(state, prediction_year_int)] = \
                                {'X_test': X_test[index_cur_state, -self.number_influenza_state_columns:],
                                 'y_test': y_test[index_cur_state, :],
                                 'predictions': [current_prediction, None]}
                        elif week_index == 0 and index_wave_threshold == 1:
                            output_by_year_state_dict[(state, prediction_year_int)]['predictions'][
                                1] = current_prediction
                        else:
                            helper_past_prediction = \
                            output_by_year_state_dict[(state, prediction_year_int)]['predictions'][index_wave_threshold]
                            output_by_year_state_dict[(state, prediction_year_int)]['predictions'][
                                index_wave_threshold] = \
                                np.hstack([helper_past_prediction, current_prediction])

                    # Constructing the first entry of the tuple output_by_week_list.

                    # Checking if the classifier has the predict_proba() method otherwise predict() is called.
                    try:
                        y_test_pred_proba_ndar = classification_pipeline.predict_proba(X_test)[:, 1]
                    except AttributeError:
                        y_test_pred_proba_ndar = classification_pipeline.predict(X_test)

                    cur_week_thresh_output_list.append((y_test_pred_proba_ndar,
                                                        y_test_current_classification_ndarray))

        return output_by_week_list, output_by_year_state_dict, output_temporal_dict


#################################
# Store, Format, Evaluate Results
#################################

def save_formatted_pred_probas_and_actual_values(states_list=['all'], no_2009_bool=False,
                                                 only_seasonal_weeks_bool=False, weeks_in_advance_int=11,
                                                 wave_thresholds=[0.8, 7.0],
                                                 feature_week_period=12,
                                                 classification_pipeline_per_week_list=[Pipeline(steps=[('preprocessing',
                                                                                          StandardScaler()),
                                                                                         ('regressor', MLPClassifier(
                                                                                             hidden_layer_sizes=
                                                                                             (20, 20, 10),
                                                                                             alpha=1,
                                                                                             learning_rate='adaptive',
                                                                                             batch_size=9000,
                                                                                             random_state=1341,
                                                                                             max_iter=1000))])]*11,
                                                 excluded_features_list=['prec_', 'humid_', 'temp_', '12', '11'],
                                                 file_str=''):
    """
    This function calls get_formatted_pred_probas_and_actual_values saves the returned 3-tuple into .pkl files
    and returns the 3-tuple.

    :param states_list: A list[str], of state names which are not removed from the data set.
    :param no_2009_bool: A bool, indicating whether the outlier year 2009 in which the swine flu occurred should be
    excluded.
    :param only_seasonal_weeks_bool: A bool, indicating whether only data from the cold weather season of a year should
    be included.
    :param weeks_in_advance_int: An int, the number of forecasting weeks.
    :param wave_thresholds: A list[float], of two thresholds. If the y values (number of reported influenza infections)
    crosses this threshold the y value is classified as 1 other wise as 0.
    :param proba_thresholds: A list[float], of two probability thresholds. Each probability threshold corresponds to a
    threshold and indicates from what prediction probability onward a 1 is predicted.
    :param feature_week_period: An int, indicating for how many weeks in the past the features are considered.
    :param classification_pipeline_per_week_list: A list, of classification pipelines or models each entry corresponds
    to one forecasting distance.
    :param excluded_features_list: A list[str], containing strs which lead to excluding a column with a name containing
    str.
    :param file_str: A str, specifying the endings of the file names to which the returned values are saved.
    :return: A 3-tuple, containing one list and two dicts. The list contains the actual values and the predictions
    probabilities grouped by week of forecast then by threshold and then by test(more precisely validation) year. The
    first dictionary groups test (more precisely validation) data and predictions by state and test (more precisely
    valdiation) year. The second dictionary provides temporal data.
    """
    if weeks_in_advance_int != len(classification_pipeline_per_week_list):
        raise ValueError('The length of the list of classification pipelines has to be equal to the weeks in advance '
                         'parameter.')

    if not re.search('[^a-zA-Z0-9]', file_str) is None:
        raise ValueError('The file str is only allowed be empty or to contain the chars a-zA-Z0-9.')

    data_frame_provider = DataFrameProvider()
    forecast_provider = ForecastProvider(data_frame_provider=data_frame_provider, states_list=states_list,
                                         no_2009_bool=no_2009_bool, only_seasonal_weeks_bool=only_seasonal_weeks_bool,
                                         weeks_in_advance_int=weeks_in_advance_int, wave_thresholds=wave_thresholds,
                                         feature_week_period=feature_week_period,
                                         classification_pipeline_per_week_list=classification_pipeline_per_week_list,
                                         excluded_features_list=excluded_features_list)

    output_by_week_list, output_by_year_state_dict, output_temporal_dict = \
        forecast_provider.get_formatted_pred_probas_and_actual_values()

    with open(
            os.path.join(os.path.dirname(__file__), r'Data\Results\influenzaDataForMetric' + file_str + '.pkl'),
            'wb') as file:
        pickle.dump(output_by_week_list, file)

    with open(os.path.join(os.path.dirname(__file__),
                           r'Data\Results\influenzaDataForAnimation' + file_str + '.pkl'), 'wb') as file:
        pickle.dump(output_by_year_state_dict, file)

    with open(os.path.join(os.path.dirname(__file__),
                           r'Data\Results\temporalDataForAnimnation' + file_str + '.pkl'), 'wb') as file:
        pickle.dump(output_temporal_dict, file)

    return output_by_week_list, output_by_year_state_dict, output_temporal_dict


def format_save_input_data_for_animation(input_by_year_state_dict, input_temporal_dict, proba_threshold1=0.5,
                                         proba_threshold2=0.5, height_threshold1=0.8, height_threshold2=7.0):
    """
    This function transforms the two input dicts into a format convenient for the webapp animation. Finally the
    formatted data is saved to .csv files.

    :param input_by_year_state_dict: A dict, containing features, target values and predictions of the test (validation)
    sets.
    :param input_temporal_dict: A dict, containing temporal information about the test (validation) set.
    :param proba_threshold1: A float, the prediction probability threshold for the first influenza threshold specifying
    the value above which a 1 is predicted.
    :param proba_threshold2: A float, the prediction probability threshold for the second influenza threshold specifying
    the value above which a 1 is predicted.
    :param height_threshold1: A float, the height of the threshold1 curve used in the d3 visualization.
    :param height_threshold1: A float, the height of the threshold2 curve used in the d3 visualization.
    :return: None
    """
    complete_unique_year_week_list = list(input_temporal_dict['complete_year_week_horizon'])

    # Extending the year week list by 10 years to cover the feature horizon.
    if complete_unique_year_week_list[0][1] < 12 or 35 < complete_unique_year_week_list[-1][1]:
        raise ValueError(
            'The first week of the year week list should be greater than 11. And the last week of the year week list should be smaller than 35')
    first_week = complete_unique_year_week_list[0][1]
    first_year = complete_unique_year_week_list[0][0]
    last_week = complete_unique_year_week_list[-1][1]
    last_year = complete_unique_year_week_list[-1][0]
    for index in range(11):
        complete_unique_year_week_list = [(first_year, first_week - 1 - index)] + complete_unique_year_week_list
    for index in range(16):
        complete_unique_year_week_list.append((last_year, last_week + 1 + index))

    # Converting the list of year, week tuples of the combined training and test period.
    # The conversion result is a corresponding list with str elements in the "yyyy-mm-dd" (e.g. "2005-05-17") format.
    # Each date corresponds to the first day of the week in the year specified by the year, week tuple.
    first_day_train_test_year_week_list_of_str = [Week(year_week[0], year_week[1]).monday().strftime("%Y-%m-%d") for
                                                  year_week in
                                                  complete_unique_year_week_list]

    for year_index, prediction_year_int in enumerate([year for year in range(2005, 2015) if year != 2009]):

        test_year_week_per_row_list = input_temporal_dict['year_week_row_by_year'][year_index]
        number_of_test_rows_int = len(test_year_week_per_row_list)

        # Getting the index of the year week of the first row in our test period with respect to the combined training
        # and test period year week list.
        index_complete_year_week_list = 0
        for index, year_week in enumerate(complete_unique_year_week_list):
            if year_week == test_year_week_per_row_list[0]:
                index_complete_year_week_list = index
                break

        # Getting the converted year week list of the test period.
        first_day_test_year_week_list_of_str = first_day_train_test_year_week_list_of_str[
                                               index_complete_year_week_list: index_complete_year_week_list + number_of_test_rows_int]

        # The date lists containing arrays. The elements of the arrays are of type str representing the start of a
        # calender week  (ISO 8601).
        past_and_future_horizon_per_row_dates_list_of_ndarrays = []
        future_horizon_per_row_dates_list_of_ndarrays = []

        num_of_feature_weeks = input_by_year_state_dict[('Bayern', prediction_year_int)]['X_test'].shape[1]
        num_of_future_weeks = input_by_year_state_dict[('Bayern', prediction_year_int)]['y_test'].shape[1]
        num_of_future_weeks_prediction = \
        input_by_year_state_dict[('Bayern', prediction_year_int)]['predictions'][0].shape[1]

        # Now we are going to iterate over each test set row.
        for row_index in range(number_of_test_rows_int):
            # For each row in the test set the current past and future dates are appended to the corresponding lists.
            past_and_future_horizon_per_row_dates_list_of_ndarrays.append(first_day_train_test_year_week_list_of_str[
                                                                          index_complete_year_week_list -
                                                                          num_of_feature_weeks: index_complete_year_week_list +
                                                                                                num_of_future_weeks + 1])
            future_horizon_per_row_dates_list_of_ndarrays.append(first_day_train_test_year_week_list_of_str[
                                                                 index_complete_year_week_list: index_complete_year_week_list +
                                                                                                num_of_future_weeks_prediction + 1])

            # Updating the index of the current week with respect to the combined training and
            # test horizon year week list.
            index_complete_year_week_list += 1

        # Store temporal data per year as csv.

        data_directory_fpslide = os.path.join(os.path.dirname(__file__), r'Data\Fpsslide')

        file_names_temporal = ['wholeDateLine' + str(prediction_year_int) + '.csv',
                               'currentDates' + str(prediction_year_int) + '.csv',
                               'futureDates' + str(prediction_year_int) + '.csv']
        to_convert_list_temporal = [first_day_test_year_week_list_of_str,
                                    past_and_future_horizon_per_row_dates_list_of_ndarrays,
                                    future_horizon_per_row_dates_list_of_ndarrays]

        for index in range(len(file_names_temporal)):
            cur_list = to_convert_list_temporal[index]
            if type(cur_list[0]) != type(""):
                pd.DataFrame(cur_list, columns=range(len(cur_list[0]))) \
                    .to_csv(os.path.join(data_directory_fpslide, file_names_temporal[index]), index=False)
            else:
                pd.DataFrame(cur_list, columns=[0]) \
                    .to_csv(os.path.join(data_directory_fpslide, file_names_temporal[index]), index=False)

        for state in all_states_list:
            # These lists contain arrays of the actual and predicted numbers of influenza infections.
            # The predictions simply state if a certain thershold of reported numbers is crossed.
            actual_influenza_numbers_list_of_ndarrays = []
            prediction1_influenza_numbers_list_of_ndarrays = []
            prediction2_influenza_numbers_list_of_ndarrays = []

            X_influenza_test = input_by_year_state_dict[(state, prediction_year_int)]['X_test']
            y_test = input_by_year_state_dict[(state, prediction_year_int)]['y_test']
            y_pred_ndarray_list = input_by_year_state_dict[(state, prediction_year_int)]['predictions']

            # Converting prediction probabilities to the y coordinates used for a visualization.
            y_pred_ndarray_threshold1 = (
                                                proba_threshold1 < np.hstack(
                                            [y_pred_ndarray_list[0], y_pred_ndarray_list[0][:, -1]
                                            .reshape(-1, 1)])).astype(int) * height_threshold1
            y_pred_ndarray_threshold2 = (
                                                proba_threshold2 < np.hstack(
                                            [y_pred_ndarray_list[1], y_pred_ndarray_list[1][:, -1]
                                            .reshape(-1, 1)])).astype(int) * height_threshold2

            for row_index in range(number_of_test_rows_int):
                # Again for each row the current actual influenza numbers as well as the to predictions are appended to
                # the corresponding lists.
                actual_influenza_numbers_list_of_ndarrays.append(
                    np.hstack([X_influenza_test[row_index, :], y_test[row_index, :], [y_test[row_index, -1]]]))
                prediction1_influenza_numbers_list_of_ndarrays.append(y_pred_ndarray_threshold1[row_index, :])
                prediction2_influenza_numbers_list_of_ndarrays.append(y_pred_ndarray_threshold2[row_index, :])

            # Store to csv per year and state.
            file_names_influenza_list = ['influenza' + state + str(prediction_year_int) + '.csv', 'prediction1' +
                                         state + str(prediction_year_int) + '.csv', 'prediction2' + state +
                                         str(prediction_year_int) + '.csv']
            to_convert_list_influenza = [actual_influenza_numbers_list_of_ndarrays,
                                         prediction1_influenza_numbers_list_of_ndarrays,
                                         prediction2_influenza_numbers_list_of_ndarrays]

            for index in range(len(file_names_influenza_list)):
                cur_list = to_convert_list_influenza[index]
                cur_num_columns = cur_list[0].shape[0]
                pd.DataFrame(cur_list, columns=range(cur_num_columns)) \
                    .to_csv(os.path.join(data_directory_fpslide, file_names_influenza_list[index]), index=False)


def visualize_and_save_metrics(week_threshold_years_list, proba_threshold_list=[0.5, 0.5], threshold_list=[0.8, 7.0],
                               threshold_color_list=["#036564", "#550b1d"], save_bool=False, file_str=''):
    """
    This function visualized the following metric values 'Accuracy', 'Precision', 'Recall', 'F2 Score', 'Log Loss',
    'AUC' and the confusion matrix via bokeh plots. Further the figures are saved to .pkl files in an embeddable format.

    :param week_threshold_years_list: A list, the target variables and predictions by week, threshold and prediction year.
    :param proba_threshold_list: A list[float], of probability thresholds.
    :param threshold_list: A list[float], of influenza number thresholds.
    :param threshold_color_list: A list[str], of  hexadecimal color values.
    :param file_str: A str, the post fix of the filenames to which the plots are saved.
    :param save_bool: A bool, specifying whether the plots are saved.
    :return: A 3-tuple, each entry is a bokeh.Row
    """
    if not re.search('[^a-zA-Z0-9]', file_str) is None:
        raise ValueError('The file_str parameter is only allowed to be empty or to contain the chars a-zA-Z0-9.')

    metric_list = [accuracy_score, precision_score, recall_score, lambda x, y: fbeta_score(x, y, 2.0), log_loss,
                   roc_auc_score]
    needs_binary_pred_bool_list = [True, True, True, True, False, False]
    metric_needs_one_actual_bool_list = [False, False, True, True, True, True]
    metric_name_list = ['Accuracy', 'Precision', 'Recall', 'F2 Score', 'Log Loss', 'AUC']
    title_metric_plot_list = [metric + " per Validation Year" for metric in metric_name_list]

    metric_plot_list = []
    metricPlotLists = [[], ([], [])]

    for index in range(len(metric_list)):
        metric_dict = format_pred_probas_and_act_values_by_week_and_threshold(week_threshold_years_list, metric=metric_list[index], metric_needs_binary_predictions_bool=
        needs_binary_pred_bool_list[index], metric_needs_one_actual_bool=metric_needs_one_actual_bool_list[index],
                                                                              proba_threshold1=proba_threshold_list[0], proba_threshold2=proba_threshold_list[1])

        if index == 0:

            for index_threshold in range(2):
                plot_list = []
                is_threshold_one = True
                if index_threshold == 1:
                    is_threshold_one = False

                for index_week in range(len(metric_dict['sorted_unique_week_list'])):
                    plot_list.append(visualize_confusion_matrix(metric_dict['all_true_values_ndarray_per_week_threshold_tuple'][index_threshold][index_week],
                                                                metric_dict['all_predictions_ndarray_per_week_threshold_tuple'][index_threshold][index_week],
                                                                title='Week ' + str(index_week + 1), threshold_proba=proba_threshold_list[index_threshold], is_threshold_one=is_threshold_one))

                metricPlotLists[1][index_threshold].extend(plot_list)

        metric_plot_list.append(visualize_metrics(metric_dict['week_str_list'], metric_dict['metric_value_list'], metric_dict['year_list'], metric_dict['threshold_list'],
                                                  metric_dict['overall_metric_per_week_and_threshold_tuple'][0], metric_dict['overall_metric_per_week_and_threshold_tuple'][1],
                                                  metric_dict['sorted_unique_week_list'], threshold1=threshold_list[0], threshold2=threshold_list[1], title=title_metric_plot_list[index],
                                                  y_axis_label=metric_name_list[index], metric_name=metric_name_list[index], color_threshold1=threshold_color_list[0],
                                                  color_threshold2=threshold_color_list[1]))

    metricPlotLists[0].extend(metric_plot_list)

    # Saving plots to pkl file if save_bool==True.
    if save_bool:
        # Building confusion rows:
        for index_threshold in range(2):
            conf_mat_plot_list = metricPlotLists[1][index_threshold]
            half_of_plot_list_length = len(conf_mat_plot_list)//2
            column_conf_mat_plot = column([row(conf_mat_plot_list[:half_of_plot_list_length]),
                                           row(conf_mat_plot_list[half_of_plot_list_length:])])

            script, div = components(column_conf_mat_plot)

            with open(os.path.join(os.path.dirname(__file__), r'Data\Plots\confMatPlotTh' + str(index_threshold + 1) + file_str + '.pkl'), 'wb') as file:
                pickle.dump((script, div), file)

        for index_metric, name_metric in enumerate(metric_name_list):
            with open(os.path.join(os.path.dirname(__file__), r'Data\Plots\metric_' + name_metric.replace(" ", "") + file_str + '_script_div.pkl'), 'wb') as file:
                pickle.dump(components(metricPlotLists[0][index_metric]), file)

    return row(metricPlotLists[0]), row(metricPlotLists[1][0]), row(metricPlotLists[1][1])


def format_pred_probas_and_act_values_by_week_and_threshold(week_threshold_years_list, metric=roc_auc_score,
                                                            threshold1=0.8, threshold2=7.0, proba_threshold1=0.5,
                                                            proba_threshold2=0.5, metric_needs_one_actual_bool=True,
                                                            metric_needs_binary_predictions_bool=True):
    """
    This function provides the evaluation results with respect to the specified metric. First, the results are provided
    by test (validation) year in the form of the first 4 lists. Second the first 2-tuple contains the metric value from
    evaluating the validation predictions from all validation folds grouped by forecasting week and threshold. The final
    two tuples contain the actual values and the predicted values also grouped by forecasting week and threshold.

    :param week_threshold_years_list: A list, containing actual and predicted values grouped by year, threshold and
    year.
    :param metric: roc_auc_score, log_loss, accuracy_score, recall_score, precision_score or fbeta, the metric with
    respect to which the predictions are evaluated.
    :param threshold1: A float, the first influenza threshold.
    :param threshold2: A float, the second influenza threshold.
    :param proba_threshold1: A float, the prediction probability threshold for the first influenza threshold specifying
    the value above which a 1 is predicted.
    :param proba_threshold2: A float, the prediction probability threshold for the second influenza threshold specifying
    the value above which a 1 is predicted.
    :param metric_needs_one_actual_bool: A bool, specifying whether the metric requires that both classes exist
    in the target value array.
    :param metric_needs_binary_predictions_bool: A bool, specifying whether the metric requires class predictions
    instead of prediction probabilities.
    :return: A dict, with values as described above.
    """

    week_list = []
    threshold_list = []
    year_list = []

    actual_values_list = []
    predictions_list = []

    metric_value_list = []
    has_ones_year_bool_list = []

    for per_week_dict in week_threshold_years_list:
        for per_threshold_dict in per_week_dict['output_by_threshold']:

            if per_threshold_dict['threshold'] != threshold1 and per_threshold_dict['threshold'] != threshold2:
                raise ValueError('A threshold of the loaded file has an unexpected value.')

            current_year = 2005
            for pred_proba_true_tuple in per_threshold_dict['output_per_year']:

                if current_year != 2005:

                    if pred_proba_true_tuple[1].sum() != 0:
                        has_ones_year_bool_list.append(True)
                    else:
                        has_ones_year_bool_list.append(False)

                    week_list.append(per_week_dict['week'])  # week'] + 1 depends on week_threshold_years_list format
                    threshold_list.append(per_threshold_dict['threshold'])
                    year_list.append(current_year)
                    actual_values_list.append(pred_proba_true_tuple[1].flatten())
                    predictions_list.append(pred_proba_true_tuple[0])

                    current_correct_format_predictions_ndarray = pred_proba_true_tuple[0]

                    if metric_needs_binary_predictions_bool:
                        if per_threshold_dict['threshold'] == threshold1:
                            current_proba_threshold = proba_threshold1
                        else:
                            current_proba_threshold = proba_threshold2

                        current_correct_format_predictions_ndarray = \
                            (current_proba_threshold <= current_correct_format_predictions_ndarray).astype(int)

                    if pred_proba_true_tuple[1].sum() != 0 or not metric_needs_one_actual_bool:
                        metric_value_list.append(
                            metric(pred_proba_true_tuple[1], current_correct_format_predictions_ndarray))
                    else:
                        metric_value_list.append(None)

                if current_year != 2008:
                    current_year += 1
                else:
                    current_year += 2

    # The unique list of weeks for the x_range of the figure.
    sorted_unique_week_list = sorted(list(set(week_list)))
    sorted_unique_week_list = ['Week ' + str(week) for week in sorted_unique_week_list]

    # Concatenating the true and predicted values per threshold and week.
    all_predictions_ndarray_per_week_threshold1 = get_true_or_predicted_values_per_week(predictions_list,
                                                                                        threshold_list, week_list,
                                                                                        sorted_unique_week_list,
                                                                                        threshold1)
    all_true_values_ndarray_per_week_threshold1 = get_true_or_predicted_values_per_week(actual_values_list,
                                                                                        threshold_list, week_list,
                                                                                        sorted_unique_week_list,
                                                                                        threshold1)
    all_predictions_ndarray_per_week_threshold2 = get_true_or_predicted_values_per_week(predictions_list,
                                                                                        threshold_list,
                                                                                        week_list,
                                                                                        sorted_unique_week_list,
                                                                                        threshold2)
    all_true_values_ndarray_per_week_threshold2 = get_true_or_predicted_values_per_week(actual_values_list,
                                                                                        threshold_list, week_list,
                                                                                        sorted_unique_week_list,
                                                                                        threshold2)

    # Calculating the scores per threshold and week.
    score_per_week_threhold1 = []
    score_per_week_threhold2 = []

    for index in range(len(sorted_unique_week_list)):
        formatted_predictions1_ndarray = all_predictions_ndarray_per_week_threshold1[index]
        formatted_predictions2_ndarray = all_predictions_ndarray_per_week_threshold2[index]
        if metric_needs_binary_predictions_bool:
            formatted_predictions1_ndarray = (proba_threshold1 <= formatted_predictions1_ndarray).astype(int)
            formatted_predictions2_ndarray = (proba_threshold2 <= formatted_predictions2_ndarray).astype(int)
        score_per_week_threhold1.append(metric(all_true_values_ndarray_per_week_threshold1[index],
                                               formatted_predictions1_ndarray))
        score_per_week_threhold2.append(metric(all_true_values_ndarray_per_week_threshold2[index],
                                               formatted_predictions2_ndarray))

    # The str week list for plotting the x_axis.
    week_str_list = ['Week ' + str(week) for week in week_list]

    if metric_needs_one_actual_bool:
        # Plotting the different AUC values for each week, threshold and test year.
        week_str_list = np.array(week_str_list)[has_ones_year_bool_list]
        metric_value_list = np.array(metric_value_list)[has_ones_year_bool_list]
        year_list = np.array(year_list)[has_ones_year_bool_list]
        threshold_list = np.array(threshold_list)[has_ones_year_bool_list]

    # Calculating the overall metric scores for the different years.
    overall_metric_per_week_and_threshold1 = score_per_week_threhold1

    # At the moment not used
    # get_metric_average_over_years_per_week(auc_list, auc_threshold_list, auc_week_list,
    # sorted_unique_week_list,
    # 0.8)

    overall_metric_per_week_and_threshold2 = score_per_week_threhold2

    # At the moment not used
    # get_metric_average_over_years_per_week(auc_list, auc_threshold_list, auc_week_list,
    # sorted_unique_week_list,
    # 7.0)

    return {'week_str_list': week_str_list, 'metric_value_list': metric_value_list, 'year_list': year_list,
            'threshold_list': threshold_list, 'overall_metric_per_week_and_threshold_tuple':
                (overall_metric_per_week_and_threshold1, overall_metric_per_week_and_threshold2),
            'sorted_unique_week_list': sorted_unique_week_list, 'all_true_values_ndarray_per_week_threshold_tuple':
                (all_true_values_ndarray_per_week_threshold1, all_true_values_ndarray_per_week_threshold2),
            'all_predictions_ndarray_per_week_threshold_tuple': (all_predictions_ndarray_per_week_threshold1,
                                                                 all_predictions_ndarray_per_week_threshold2)}


def get_metric_average_over_years_per_week(metric_value_list, threshold_list, week_list, sorted_unique_week_list,
                                           threshold):
    """
    This function returns for each forecasting distance the average over the per validation year metric values. Each of
    the first three parameter lists are associated via the index. So, for instance metric_value_list[1] contains the
    metric value for one validation years for the threshold threshold_list[1] and the forecasting week week_list[1].

    :param metric_value_list: A list[float], containing the metric values.
    :param threshold_list: A list[float], containing the thresholds for the reported number of influenza cases.
    :param week_list: A list[int], containing the weeks.
    :param sorted_unique_week_list: A list[int], a sorted list of the forecasting distances in weeks.
    :param threshold: A float, the threshold of interest.
    :return: A list[float], containing the averaged metric score per forecasting week.
    """
    # Calculating the average per week and threshold.
    sum_per_week_and_threshold = [0 for _ in sorted_unique_week_list]
    count_per_week_and_threshold = [0 for _ in sorted_unique_week_list]

    for index in range(len(threshold_list)):
        if threshold_list[index] == threshold:
            sum_per_week_and_threshold[week_list[index] - 1] += metric_value_list[index]
            count_per_week_and_threshold[week_list[index] - 1] += 1

    return np.array(sum_per_week_and_threshold) / np.array(count_per_week_and_threshold)


def get_true_or_predicted_values_per_week(value_list, threshold_list, week_list, sorted_unique_week_list, threshold):
    """
    This function returns for each forecasting distance the concatenation over the per validation year values (for
    instance the actual values or the predicted values) provided via value_list. Each of the first three parameter lists
    are associated via the index. So for instance values_list[1] contains the values for a validation year associated to
    the threshold threshold_list[1] and the forecasting week week_list[1].

    :param value_list: A list[numpy.ndarray], containing the values (either actual target values or predicted values).
    :param threshold_list:  A list[float], containing the thresholds for the reported number of influenza cases.
    :param week_list: A list[int], containing the weeks.
    :param sorted_unique_week_list: A list[int], a sorted list of the forecasting distances in weeks.
    :param threshold: A float, the threshold of interest.
    :return: A list[numpy.ndarray], containing the concatenated values per forecasting week.
    """
    value_ndarray_per_week_list = [None for _ in sorted_unique_week_list]

    for index in range(len(threshold_list)):
        if threshold_list[index] == threshold:
            if value_ndarray_per_week_list[week_list[index]-1] is None:
                value_ndarray_per_week_list[week_list[index]-1] = value_list[index]
            else:
                value_ndarray_per_week_list[week_list[index]-1] = np.hstack([value_ndarray_per_week_list[week_list[index]-1], value_list[index]])

    return value_ndarray_per_week_list


def visualize_metrics(week_str_list, metric_value_list, year_list, threshold_list,
                      overall_metric_per_week_and_threshold1, overall_metric_per_week_and_threshold2,
                      sorted_unique_week_list, threshold1=0.8, threshold2=7.0, title='', y_axis_label='Metric Value',
                      color_threshold1="#036564", color_threshold2="#550b1d", metric_name='Metric'):


    """
    This function visualized the metric values. On the one hand it visualized the metric values per validation year.
    On the other hand it visualized the metric when evaluated on all validation years simultaneously.

    :param week_str_list: A list[str], containing the weeks.
    :param metric_value_list: A list[float], containing the metric values.
    :param year_list: A list[int], of the validation years.
    :param threshold_list: A list[float], containing the classification thresholds for the reported number of influenza
    cases.
    :param overall_metric_per_week_and_threshold1:
    :param overall_metric_per_week_and_threshold2:
    :param sorted_unique_week_list:
    :param threshold1: A float, the first classification threshold.
    :param threshold2: A float, the second classification threshold.
    :param title: A str, the title of the figure.
    :param y_axis_label: A str, the label of the y-axis.
    :param color_threshold1: A str, the color associated to the first threshold.
    :param color_threshold2: A str, the color associated to the first threshold.
    :param metric_name: A str, the name of the metric.
    :return: A bokeh figure, visualizing the metric values.
    """
    TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"

    p = figure(title=title, x_range=sorted_unique_week_list, x_axis_label='Forecasting Distance',
               y_axis_label=y_axis_label, plot_width=600, plot_height=500, tools=TOOLS)

    # Plotting metric data per year for two thresholds.
    index_bool_threshold1_ndarray = np.array(threshold_list) == threshold1
    index_bool_threshold2_ndarray = np.logical_not(index_bool_threshold1_ndarray)

    index_list = [index_bool_threshold1_ndarray, index_bool_threshold2_ndarray]
    color_list = [color_threshold1, color_threshold2]
    thresh_list = [threshold1, threshold2]

    for index in range(2):
        week_str_ndarray = np.array(week_str_list)[index_list[index]]

        metric_value_ndarray = np.array(metric_value_list)[index_list[index]]

        year_ndarray = np.array(year_list)[index_list[index]]

        threshold_ndarray = np.array(threshold_list)[index_list[index]]

        metric_data_dict = dict(x=week_str_ndarray, y=metric_value_ndarray, year=year_ndarray,
                                threshold=threshold_ndarray)

        metric_source = ColumnDataSource(data=metric_data_dict)

        p.x(x='x', y='y', size=10, color=color_list[index], source=metric_source,
            legend='Per Year ' + metric_name + ': Threshold ' + str(thresh_list[index]), muted_color=color_list[index],
            muted_alpha=0.0)

    # Plotting the Average over the metric values for the different years.
    average1_metric_source = ColumnDataSource(data=dict(x=sorted_unique_week_list, y=overall_metric_per_week_and_threshold1,
                                                     year=['2005, ..., 2008, 2010, ... 2014'] * len(sorted_unique_week_list),
                                                     threshold=['0.8'] * len(sorted_unique_week_list)))
    average2_metric_source = ColumnDataSource(data=dict(x=sorted_unique_week_list, y=overall_metric_per_week_and_threshold2,
                                                     year=['2005, ..., 2008, 2010, ... 2014'] * len(
                                                         sorted_unique_week_list),
                                                     threshold=['7.0'] * len(sorted_unique_week_list)))
    p.rect(x='x', y='y', width=0.8, height=5, source=average1_metric_source, color=color_threshold1,
           muted_color=color_threshold1, muted_alpha=0.0, legend=metric_name + ' Overall: Threshold ' + str(threshold1),
           height_units="screen")
    p.rect(x='x', y='y', width=0.8, height=5, source=average2_metric_source, color=color_threshold2,
           muted_color=color_threshold2, muted_alpha=0.0, legend=metric_name + 'Overall: Threshold ' + str(threshold2),
           height_units="screen")

    # Some formatting and interaction
    p.xaxis.major_label_orientation = 3.0 / 4
    p.legend.location = "bottom_left"
    p.legend.background_fill_alpha = 0.0
    p.legend.click_policy = "mute"

    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [("Year(s)", "@year")]

    return p


def visualize_confusion_matrix(actual_classes_ndarray, predicted_probabilities_ndarray, threshold_proba=0.05, title='',
                               is_threshold_one=True):
    """
    This function visualized the confusion matrix via bokeh figures.

    :param actual_classes_ndarray: A numpy.ndarray[int], the actual target values.
    :param predicted_probabilities_ndarray: A numpy.ndarray[float], the prediction probabilities.
    :param threshold_proba: A float, the probability threshold specifying from when on a class is classified as 1.
    :param title: A str, the title of the figure.
    :param is_threshold_one: A bool, specifying whether the first or the second classification threshold is considered.
    :return: A bokeh figure, visualizing the confusion matrices.
    """

    true_values_ndarray_per_week = actual_classes_ndarray
    predicted_values_ndarray_per_week = (threshold_proba <= predicted_probabilities_ndarray).astype(int)

    cur_confusion_matrix = confusion_matrix(true_values_ndarray_per_week, predicted_values_ndarray_per_week)

    true_negatives = cur_confusion_matrix[0, 0]
    false_positives = cur_confusion_matrix[0, 1]
    false_negatives = cur_confusion_matrix[1, 0]
    true_positives = cur_confusion_matrix[1, 1]

    # Categories for the x and y-axis.
    labels_prediction = ('Pred. Class 0', 'Pred. Class 1')
    labels_truth = ('Class 0', 'Class 1')

    start_length_count_df = pd.DataFrame(
        [[labels_prediction[0], labels_truth[0], true_negatives], [labels_prediction[1], labels_truth[0], false_positives],
         [labels_prediction[0], labels_truth[1], false_negatives],
         [labels_prediction[1], labels_truth[1], true_positives]], columns=['predicted_class', 'true_class', 'count'])

    # Coloring according to threshold
    if is_threshold_one:
        colors = ["#f4f9f8", "#daeae4", "#ceefea", "#a8c4c0", "#829995", "#546360"]
    else:
        colors = ["#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41"]

    mapper = LogColorMapper(palette=colors, low=0, high=7000)

    source = ColumnDataSource(start_length_count_df)

    p = figure(title=title, x_range=labels_prediction, y_range=list(labels_truth)[::-1],
               plot_width=145, plot_height=145,
               x_axis_location="above")

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "6pt"
    p.axis.major_label_standoff = 0

    p.rect(x='predicted_class', y='true_class', width=1, height=1,
           source=source,
           fill_color={'field': 'count', 'transform': mapper},
           line_color=None)

    p.toolbar.logo = None
    p.toolbar_location = None

    labels = LabelSet(x='predicted_class', y='true_class', text='count', level='glyph',
                      x_offset=0, y_offset=0, source=source, render_mode='canvas', text_align="center",
                      text_baseline="middle", text_color="black", text_font_size="8pt")

    p.add_layout(labels)

    return p