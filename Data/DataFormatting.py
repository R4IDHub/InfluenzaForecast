"""
@author: Anton Sporrer, anton.sporrer@yahoo.com
"""

# Basic imports
import os
import re
from datetime import date
import pandas as pd

# Storing and fetching data
import dill as pickle

# Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

"""
This module consists of the DataFrameProvider class. This class loads and transforms data. This module further provides 
functions for reformatting data frames beyond the initial transformations.
"""

class DataFrameProvider(object):
    """
    This class implements the formatting of the data stored in .xlsx files.
    Namely the reported number of influenza infections, the google trends data and
    weather data - all on a state level.

    The separate files are formatted into two Pandas dataframes self.rawXDataFrame, self.rawYDataFrame.
    The format specifics of the format is defined by the private instance variables. E.g. __lagToCurrentWeek,
    ... , __weatherWeeks.

    The X dataframe has a rolling window format. That is to say each row contains the influenza numbers,
    google trends data and weather data of the past weeks. Such a row is provided for each week of in the time
    horizon and for each state separately. This row will be feed into for instance a neural net and then the
    actual number of influenza cases of the next week will be forcasted.

    The Y data frame contains the associated number of influenza cases of a future point in time. In other
    words Y contains the target variables.

    The method :func:`~DataFormatting.DataFrameProvider.getFeaturesDF` provides the X dataframe to the user of
    this class. Final specifications can be made at this point.
    """

    def __init__(self, lagToCurrentWeek=1, influenzaPerStateWeeks=12, influenzaGermanyWeeks=12, \
                                trendPerStateWeeks=12, trendGermanyWeeks=12, weatherWeeks=12):
        """
        If the constructor parameters correspond to a previously stored dataframes. The
        previous dataframes is loaded. Otherwise the method :func:`~DataFormatting.DataFrameProvider.__createNewRawDataFrame`
        is called and new raw dataframes with the specified format are constructed.

        :param lagToCurrentWeek: An int, the forcasting period in weeks.
        :param influenzaPerStateWeeks: An int, the number of weeks of past influenza numbers per state in the rolling window.
        :param influenzaGermanyWeeks: An int, the number of weeks of past influenza numbers in Germany in the rolling window.
        :param trendPerStateWeeks:  An int, the number of weeks of past google trends data per state in the rolling window.
        :param trendGermanyWeeks:  An int, the number of weeks of past google trends data in Germany in the rolling window.
        :param weatherWeeks:  An int, the number of weeks of past weather data in the rolling window.
        """

        self.__lagToCurrentWeek = lagToCurrentWeek
        self.__influenzaPerStateWeeks = influenzaPerStateWeeks
        self.__influenzaGermanyWeeks = influenzaGermanyWeeks
        self.__trendPerStateWeeks = trendPerStateWeeks
        self.__trendGermanyWeeks = trendGermanyWeeks
        self.__weatherWeeks = weatherWeeks

        # Checking if the a dataframe has already been formatted and stored in the past.
        # If so, in the following the formatting of the stored dataframes is compared to the
        # currently desired dataframe format specified by the instance variables.
        # If stored format is equal to the desired format then the stored dataframes are simply
        # returned.
        if os.path.isfile(os.path.dirname(__file__) + r'\FormattedData\instance_variables.pkl') and \
                os.path.isfile(os.path.dirname(__file__) + r'\FormattedData\features.pkl') and \
                os.path.isfile(os.path.dirname(__file__) + r'\FormattedData\labels.pkl'):

            # Getting the instance variables.
            with open(os.path.dirname(__file__) + r'.\FormattedData\instance_variables.pkl', 'rb') as file:
                weeks_instance_variables_tuple = pickle.load(file)

            if self.__lagToCurrentWeek == weeks_instance_variables_tuple[0] and \
                self.__influenzaPerStateWeeks == weeks_instance_variables_tuple[1] and \
                self.__influenzaGermanyWeeks == weeks_instance_variables_tuple[2] and \
                self.__trendPerStateWeeks == weeks_instance_variables_tuple[3] and \
                self.__trendGermanyWeeks == weeks_instance_variables_tuple[4] and \
                self.__weatherWeeks == weeks_instance_variables_tuple[5]:

                # DataFrame containing the raw feature information for our prediction.
                with open(os.path.dirname(__file__) + r'.\FormattedData\features.pkl', 'rb') as file:
                    self.rawXDataFrame = pickle.load(file)
                # DataFrame containing the raw target variables for our prediction.
                with open(os.path.dirname(__file__) + r'.\FormattedData\labels.pkl', 'rb') as file:
                    self.rawYDataFrame = pickle.load(file)
            else:
                self.__createNewRawDataFrame()

        # Otherwise the dataframe in the desired format is recalculated.
        else:
            self.__createNewRawDataFrame()

    def __createNewRawDataFrame(self):
        """
        This method is used to create new dataframes according to the specified instance variables.
        These dataframes are assigned to the instance variables. self.rawXDataFrame and self.rawYDataFrame.
        Further they are saved to .pkl files. By saving the dataframes they can be used in different session
        without having to reformat the dataframes every time.
        """

        # Dataframes with the specified format for the influenza numbers, the google trends data and the weather data
        # are generated by the respective private class methods.
        rawInfluenzaPerStateDF = self.__calculateRawInfluenzaDataFrame()
        rawTrendPerStateDF = self.__calculateRawTrendDataFrame()
        rawWeatherStateDF = self.__calculateRawWeatherDataFrame()

        # The separate dataframes are merged.
        rawDF = pd.merge(rawWeatherStateDF, rawTrendPerStateDF, on=['year_week', 'state'], how='inner')
        rawDF = pd.merge(rawDF, rawInfluenzaPerStateDF, on=['year_week', 'state'], how='inner')

        # The instance variables are assigned.
        self.rawXDataFrame = rawDF[rawDF.columns[:-1]]
        self.rawYDataFrame = rawDF[rawDF.columns[-1]]

        # Three things are saved to .pkl files. Which can be reused in later sessions.
        # 1.) The format information stored in the instance variables.
        # 2.) The X dataframe containing the rolling window rows.
        # 3.) The Y target variables associated to the rolling window rows of X.
        with open(os.path.dirname(__file__) + r'.\FormattedData\instance_variables.pkl', 'wb') as file:
            pickle.dump((self.__lagToCurrentWeek, self.__influenzaPerStateWeeks, self.__influenzaGermanyWeeks,
                         self.__trendPerStateWeeks, self.__trendGermanyWeeks, self.__weatherWeeks), file)
        with open(os.path.dirname(__file__) + r'.\FormattedData\features.pkl', 'wb') as file:
            pickle.dump(self.rawXDataFrame, file)
        with open(os.path.dirname(__file__) + r'.\FormattedData\labels.pkl', 'wb') as file:
            pickle.dump(self.rawYDataFrame, file)

    ########################################################
    # The following three methods beginning with "__calculate..."
    # are simply reading the associated .xlsx files. Further they
    # are calling the associated "__get..." method to format the
    # the data read from the .xlsx files. This formatted data
    # is then returned.
    ########################################################

    def __calculateRawInfluenzaDataFrame(self):
        """
        The specified .xlsx file is read into a dataframe. Some reformatting is performed before
        calling the __getInfluenzaRowsDF method which does the main reformatting of the data. Reformatting
        means that the data is converted into a rolling window format.
        :return: A Pandas dataframe, the dataframe with the influenza numbers in a rolling window format.
        """
        # Loading excel file
        influenzaPerStateColumnDF = pd.read_excel(os.path.dirname(__file__) + r'/InfluenzaNumbers/InfluenzaStateLevel20012016.xlsx')

        # Empty values correspond to 0 reported cases.
        influenzaPerStateColumnDF = influenzaPerStateColumnDF.fillna(value=0)

        # Adding a year week column in the correct format and deleting the old one.
        influenzaPerStateColumnDF['year_week'] = influenzaPerStateColumnDF['Year_and_week_of_notification'].apply(
            lambda x: (int(str(x)[:4]), int(str(x)[6:])))

        influenzaPerStateColumnDF = influenzaPerStateColumnDF.drop(['Year_and_week_of_notification'], axis=1)

        # Adding the Germany column containing the reported cases in Germany per 100 000 inhabitants.
        influenzaPerStateColumnDF['Deutschland'] = influenzaPerStateColumnDF[
            influenzaPerStateColumnDF.columns[:-1]].apply(lambda x: x.mean(), axis=1)

        # Getting the state names from the DataFrame object. The last two column names are excluded. They are the year_week and 'Deutschland' column.
        state_names = influenzaPerStateColumnDF.columns.values[:-2]

        rawInfluenzaDF = None

        # The dataframes are constructed separately for each state. They are concatenated afterwards.
        for state_name in state_names:

            print('Constructing the influenza number features for the state: ' + str(state_name))

            currentStaterawInfluenzaDF = self.__getInfluenzaRowsDF(influenzaPerStateColumnDF['year_week'].tolist(), \
                                                                   influenzaPerStateColumnDF[state_name].tolist(), \
                                                                   influenzaPerStateColumnDF['Deutschland'].tolist(), \
                                                                   state_name, \
                                                                   self.__influenzaPerStateWeeks, \
                                                                   self.__influenzaGermanyWeeks, \
                                                                   self.__lagToCurrentWeek)

            if rawInfluenzaDF is not None:
                rawInfluenzaDF = pd.concat([rawInfluenzaDF, currentStaterawInfluenzaDF])
            else:
                rawInfluenzaDF = currentStaterawInfluenzaDF

        return rawInfluenzaDF

    def __calculateRawTrendDataFrame(self):
        """
        The specified .xlsx file is read into a dataframe. Some reformatting is performed before
        calling the __getTrendRowsDF method which does the main reformatting of the data. Reformatting
        means that the data is converted into a rolling window format.
        :return: A Pandas dataframe, the dataframe with the google trends data in a rolling window format.
        """

        # Loading excel file
        trendPerStateColumnDF = pd.read_excel(os.path.dirname(__file__) + r'/FluTrends/FluTrendsStateLevel.xlsx')

        # Empty values correspond to 0 reported cases.
        trendPerStateColumnDF = trendPerStateColumnDF.fillna(value=0)

        # Adding a year week column in the correct format and deleting the cold one.
        trendPerStateColumnDF['year_week'] = trendPerStateColumnDF['Date'].apply(
            lambda x: self.__convert_str_to_date(str(x)))
        trendPerStateColumnDF['year_week'] = trendPerStateColumnDF['year_week'].apply(
            lambda x: (x.isocalendar()[0], x.isocalendar()[1]))

        trendPerStateColumnDF = trendPerStateColumnDF.drop(['Date'], axis=1)

        # Getting the state names from the DataFrame object. The last first and the last column names are excluded. They are the 'Deutschland' the year_week column.
        state_names = trendPerStateColumnDF.columns.values[1:-1]

        rawTrendDF = None

        # The dataframes are constructed separately for each state. They are concatenated afterwards.
        for state_name in state_names:

            print('Constructing the google trends features for the state: ' + str(state_name))

            currentStaterawTrendDF = self.__getTrendRowsDF(trendPerStateColumnDF['year_week'].tolist(), \
                                                           trendPerStateColumnDF[state_name].tolist(), \
                                                           trendPerStateColumnDF['Deutschland'].tolist(), \
                                                           state_name, \
                                                           self.__trendPerStateWeeks, \
                                                           self.__trendGermanyWeeks, \
                                                           self.__lagToCurrentWeek)

            if rawTrendDF is not None:
                rawTrendDF = pd.concat([rawTrendDF, currentStaterawTrendDF])
            else:
                rawTrendDF = currentStaterawTrendDF

        return rawTrendDF

    def __calculateRawWeatherDataFrame(self):
        """
        The specified .xlsx file is read into a dataframe. Some reformatting is performed before
        calling the __getWeatherRowsDF method which does the main reformatting of the data. Reformatting
        means that the data is converted into a rolling window format.
        :return: A Pandas dataframe, the dataframe with the weather data in a rolling window format.
        """
        sate_names = ['Baden-Wuerttemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg', 'Hessen',
                      'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland',
                      'Sachsen', 'Sachsen-Anhalt', 'Schleswig-Holstein', 'Thueringen']

        rawWeatherDF = None

        # The dataframes are constructed separately for each state. They are concatenated afterwards.
        for state_name in sate_names:

            print('Constructing the weather features for the state: ' + str(state_name))

            current_raw_weather_df = pd.read_excel(os.path.dirname(__file__) + r'/WeatherStateLevel/' + state_name + '.xlsx')
            # Dropping the stations id column.
            current_raw_weather_df = current_raw_weather_df.drop(['STATIONS_ID'], axis=1)
            current_raw_weather_df.columns = [header.strip() for header in list(current_raw_weather_df.columns.values)]

            # Just considering dates between (inclusive) 01.01.2001 - 31.12.2016
            start_index = current_raw_weather_df[current_raw_weather_df['measurement_date'] == 20010101].index[0]
            end_index = current_raw_weather_df[current_raw_weather_df['measurement_date'] == 20161225].index[0]
            current_raw_weather_df = current_raw_weather_df.iloc[start_index:end_index + 1]

            if current_raw_weather_df.shape[0] != 5838 or current_raw_weather_df.shape[1] != 17:
                raise Exception(
                    'The number of rows of the parsed weather table should be 5838 and the number of columns should be 17.')

            # Converting the first column from string to datetime.
            current_raw_weather_df['measurement_date'] = current_raw_weather_df['measurement_date'].apply(
                lambda x: self.__convert_longdate_to_datetime(x))

            # Adding a week of the year column.
            current_raw_weather_df['year_week'] = current_raw_weather_df['measurement_date'].apply(
                lambda x: (x.isocalendar()[0], x.isocalendar()[1]))

            current_weather_df = self.__getWeatherRowsDF(current_raw_weather_df['measurement_date'].tolist(), \
                                                         current_raw_weather_df['year_week'].tolist(), \
                                                         state_name, \
                                                         current_raw_weather_df['daily_mean_temperature'].tolist(), \
                                                         current_raw_weather_df[
                                                             'daily_mean_of_relative_humidity'].tolist(), \
                                                         current_raw_weather_df[
                                                             'daily_precipitation_height_mm'].tolist(), \
                                                         self.__weatherWeeks, \
                                                         self.__lagToCurrentWeek)

            if rawWeatherDF is not None:
                rawWeatherDF = pd.concat([rawWeatherDF, current_weather_df])
            else:
                rawWeatherDF = current_weather_df

        return rawWeatherDF

    ####################################################################################
    # The following three "__get..."-methods implement the major share of the data conversion
    # into a rolling window format.
    ####################################################################################

    def __getInfluenzaRowsDF(self, yearWeekList, influenzaInStateList, influenzaGermanyList, state_name,
                             number_of_past_weeks_state, number_of_past_weeks_germany, lagToCurrentWeek):
        """
        This method uses the following lists and other parameters to construct a dataframe in the rolling window format.
        One row of the returned dataframe is one rolling window. The indices of the lists correspond to each other.
        :param yearWeekList: A list of int tuples, each tuple contains the year and the week.
        :param influenzaInStateList: A list of ints, the number of influenza cases for the current state
        :param influenzaGermanyList: A list of ints, the number of influenza cases for the Germany
        :param state_name: A string, the name of the current state.
        :param number_of_past_weeks_state: An int, the number of weeks for the influenza number window on a state level.
        :param number_of_past_weeks_germany: An int, the number of weeks for the influenza number window for Germany.
        :param lagToCurrentWeek: An int, the number of weeks forecasted in advance.
        :return: A pandas.DataFrame, in the rolling window format. Each row corresponds to one state and one week
        and contains the influenza numbers of the past weeks.
        """

        # The empty but formatted DataFrame which will be filled in the following.
        past_weeks_state_column_names = ['influenza_week-' + str(index + (lagToCurrentWeek - 1)) for index in
                                         range(number_of_past_weeks_state, 0, -1)]
        past_weeks_germany_column_names = ['influenza_germany_week-' + str(index + (lagToCurrentWeek - 1)) for index in
                                           range(number_of_past_weeks_germany, 0, -1)]

        column_names = ['year_week', 'state']
        column_names.extend(past_weeks_germany_column_names)
        column_names.extend(past_weeks_state_column_names)
        column_names.append('influenza_week_current')
        return_df = pd.DataFrame(columns=column_names)

        # The for loop starts at at the earliest possible index. Earliest possible in the sense that all weeks in the
        # associated time window are part of the data we use in this project.
        # lagToCurrentWeek - 1 since the list yearWeekList indices start at 0.
        start_index = max(number_of_past_weeks_state, number_of_past_weeks_germany) + lagToCurrentWeek - 1

        for index in range(start_index, len(yearWeekList)):
            # Creating a new row for the returned DataFrame.
            new_row = []
            # Appending the current year and week of the year.
            new_row.append(yearWeekList[index])
            # Appending the current city.
            new_row.append(state_name)

            # Appending the past number of reported cases in Germany.
            past_weeks_germany = influenzaGermanyList[index - number_of_past_weeks_germany - (
                    lagToCurrentWeek - 1): index - lagToCurrentWeek + 1]
            new_row.extend(past_weeks_germany)

            # Appending the past number of reported cases in the current state.
            past_weeks_state = influenzaInStateList[index - number_of_past_weeks_state - (
                    lagToCurrentWeek - 1): index - lagToCurrentWeek + 1]
            new_row.extend(past_weeks_state)

            # Appending the current weeks numbers.
            new_row.append(influenzaInStateList[index])

            return_df.loc[index - start_index] = new_row

        return return_df

    def __getTrendRowsDF(self, yearWeekList, trendInStateList, trendInGermanyList, state_name,
                         number_of_past_weeks_state, number_of_past_weeks_germany, lagToCurrentWeek):
        """
        This method uses the following lists and other parameters to construct a dataframe in the rolling window format.
        One row of the returned data frame is one rolling window. The indices of the lists correspond to each other.
        :param yearWeekList: A list of int tuples, each tuple contains the year and the week.
        :param trendInStateList: A list of ints, the google trends data for the current state
        :param trendInGermanyList: A list of ints, the google trends data for the Germany
        :param state_name: A string, the name of the current state.
        :param number_of_past_weeks_state: An int, the number of weeks for the google trends data window on a state level.
        :param number_of_past_weeks_germany: An int, the number of weeks for the google trends data window for Germany.
        :param lagToCurrentWeek: An int, the number of weeks forecasted in advance.
        :return: A pandas.DataFrame, in the rolling window format. Each row corresponds to one state and one week
        and contains the google trends data of the past weeks.
        """

        # The empty but formatted DataFrame which will be filled in the following.
        past_weeks_state_column_names = ['trend_week-' + str(index + (lagToCurrentWeek - 1)) for index in
                                         range(number_of_past_weeks_state, 0, -1)]
        past_weeks_germany_column_names = ['trend_germany_week-' + str(index + (lagToCurrentWeek - 1)) for index in
                                           range(number_of_past_weeks_germany, 0, -1)]

        column_names = ['year_week', 'state']
        column_names.extend(past_weeks_germany_column_names)
        column_names.extend(past_weeks_state_column_names)

        return_df = pd.DataFrame(columns=column_names)

        # The for loop starts at at the earliest possible index. Earliest possible in the sense that all weeks in the
        # associated time window are part of the data we use in this project.
        # lagToCurrentWeek - 1 since the list yearWeekList indices start at 0.
        start_index = max(number_of_past_weeks_state, number_of_past_weeks_germany) + lagToCurrentWeek - 1

        for index in range(start_index, len(yearWeekList)):
            # Creating a new row for the returned DataFrame.
            new_row = []
            # Appending the current year and week of the year.
            new_row.append(yearWeekList[index])
            # Appending the current city.
            new_row.append(state_name)

            # Appending the past number of reported cases in Germany.
            past_weeks_germany = trendInGermanyList[index - number_of_past_weeks_germany - (
                    lagToCurrentWeek - 1): index - lagToCurrentWeek + 1]
            new_row.extend(past_weeks_germany)

            # Appending the past number of reported cases in the current state.
            past_weeks_state = trendInStateList[index - number_of_past_weeks_state - (
                    lagToCurrentWeek - 1): index - lagToCurrentWeek + 1]
            new_row.extend(past_weeks_state)

            return_df.loc[index - start_index] = new_row

        return return_df

    def __getWeatherRowsDF(self, dates_list, year_week_list, state_name, temp_list, humid_list, pre_list,
                           number_of_past_weeks, lagToCurrentWeek):
        """
        This method uses the following lists and other parameters to construct a dataframe in the rolling window format.
        One row of the returned data frame is one rolling window. The indices of the lists correspond to each other.
        :param dates_list: A list of dates,
        :param year_week_list: A list of int tuples, each tuple contains the year and the week.
        :param state_name: A string, the name of the current state.
        :param temp_list: A list of ints, each int represents the temperature at the current day.
        :param humid_list: A list of ints, each int represents the humidity at the current day.
        :param pre_list: A list of ints, each int represents the precipitation at the current day.
        :param number_of_past_weeks:
        :param lagToCurrentWeek:  An int, the number of weeks forecasted in advance.
        :return:  A pandas.Dataframe, in the rolling window format. Each row corresponds to one state and one week
        and contains the weather data of the past weeks.
        """

        if len(dates_list) % 7: raise ValueError(
            'Only days of complete weeks should be in dates_list. This means dates_list % 7 == 0')

        past_days_temp_column_names = ['temp_day-' + str(index + (lagToCurrentWeek - 1) * 7) for index in
                                       range(7 * number_of_past_weeks, 0, -1)]
        past_days_humid_column_names = ['humid_day-' + str(index + (lagToCurrentWeek - 1) * 7) for index in
                                        range(7 * number_of_past_weeks, 0, -1)]
        past_days_prec_column_names = ['prec_day-' + str(index + (lagToCurrentWeek - 1) * 7) for index in
                                       range(7 * number_of_past_weeks, 0, -1)]

        column_names = ['year_week', 'state']
        column_names.extend(past_days_temp_column_names)
        column_names.extend(past_days_humid_column_names)
        column_names.extend(past_days_prec_column_names)

        # The empty but formatted DataFrame which will be filled in the following.
        return_df = pd.DataFrame(columns=column_names)

        for index in range(number_of_past_weeks + (lagToCurrentWeek - 1), len(set(year_week_list))):
            # Creating a new row for the returned DataFrame.
            new_row = []
            # Appending the current year and week of the year.
            new_row.append(year_week_list[index * 7])
            # Appending the current city.
            new_row.append(state_name)

            # The temperature, humidity, precipitation and wind in the past number_of_past_weeks * 7 days.
            past_days_temp_list = temp_list[(index - (lagToCurrentWeek - 1) - number_of_past_weeks) * 7: (index - (
                    lagToCurrentWeek - 1)) * 7]
            past_days_humid_list = humid_list[(index - (lagToCurrentWeek - 1) - number_of_past_weeks) * 7: (index - (
                    lagToCurrentWeek - 1)) * 7]
            past_days_pre_list = pre_list[(index - (lagToCurrentWeek - 1) - number_of_past_weeks) * 7: (index - (
                    lagToCurrentWeek - 1)) * 7]

            new_row.extend(past_days_temp_list)
            new_row.extend(past_days_humid_list)
            new_row.extend(past_days_pre_list)

            return_df.loc[index - number_of_past_weeks] = new_row

        return return_df

    ######################################
    # Helper functions for date conversion
    ######################################

    def __convert_longdate_to_datetime(self, date_long):
        date_string = str(date_long)
        return date(int(date_string[:4]), int(date_string[4:6]), int(date_string[6:8]))

    def __convert_str_to_date(self, date_str):
        return date(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]))

    ########################################
    # The feature data frame (pandas.DataFrame) is constructed
    ########################################

    def getFeaturesDF(self, number_of_temp_intervals=2, number_of_humid_intervals=2, number_of_prec_intervals=2,
                      encode_states=False, encode_weeks=False):
        """
        This public method provides the data frame (pandas.DataFrame) in the specified format. It uses the previously
        formatted raw data frames (instance variabiles) and does the final reformatting of the weather features. This
        final reformatting is done according to these specified parameters. For each of the intervals into which the
        temperature, humidity and precipitation data is divided the interval mean, variance, min and max is calculated
        and used as feature in the returned pandas.DataFrame object.
        :param number_of_temp_intervals: An int, the number of equally sized intervals into which the temperature data
        is divided.
        :param number_of_humid_intervals: An int, the number of equally sized intervals into which the humidity data
        is divided.
        :param number_of_prec_intervals: An int, the number of equally sized intervals into which the precipitation data
        is divided.
        :param encode_states:
        :return: A pandas.DataFrame, this data frame contains the influenza, google trends and weather features in a
        rolling window format as specified by the constructor parameters of this class and the parameters of this
        method.
        """

        # Selecting the temperature, humidity and precipitation related columns.
        temperature_df = self.rawXDataFrame[
            self.rawXDataFrame.columns[[bool(re.findall(r'temp_day', x)) for x in self.rawXDataFrame.columns]]]
        humidity_df = self.rawXDataFrame[
            self.rawXDataFrame.columns[[bool(re.findall(r'humid_day', x)) for x in self.rawXDataFrame.columns]]]
        precipitation_df = self.rawXDataFrame[
            self.rawXDataFrame.columns[[bool(re.findall(r'prec_day', x)) for x in self.rawXDataFrame.columns]]]

        # Replacing one error in data.
        precipitation_df = precipitation_df.replace(-999, 0)
        humidity_df = humidity_df.replace(-999, 8000)
        temperature_df = temperature_df.replace(-999, 10)

        # The headers of the feature columns
        temp_variance_column_names = ['temp_variance-' + str(index) for index in range(number_of_temp_intervals, 0, -1)]
        temp_mean_column_names = ['temp_mean-' + str(index) for index in range(number_of_temp_intervals, 0, -1)]
        temp_min_column_names = ['temp_min-' + str(index) for index in range(number_of_temp_intervals, 0, -1)]
        temp_max_column_names = ['temp_max-' + str(index) for index in range(number_of_temp_intervals, 0, -1)]

        humid_variance_column_names = ['humid_variance-' + str(index) for index in
                                       range(number_of_humid_intervals, 0, -1)]
        humid_mean_column_names = ['humid_mean-' + str(index) for index in range(number_of_humid_intervals, 0, -1)]
        humid_min_column_names = ['humid_min-' + str(index) for index in range(number_of_humid_intervals, 0, -1)]
        humid_max_column_names = ['humid_max-' + str(index) for index in range(number_of_humid_intervals, 0, -1)]

        prec_variance_column_names = ['prec_variance-' + str(index) for index in range(number_of_prec_intervals, 0, -1)]
        prec_mean_column_names = ['prec_mean-' + str(index) for index in range(number_of_prec_intervals, 0, -1)]
        prec_min_column_names = ['prec_min-' + str(index) for index in range(number_of_prec_intervals, 0, -1)]
        prec_max_column_names = ['prec_max-' + str(index) for index in range(number_of_prec_intervals, 0, -1)]

        # Joining the column names together
        column_names = temp_variance_column_names
        column_names.extend(temp_mean_column_names)
        column_names.extend(temp_min_column_names)
        column_names.extend(temp_max_column_names)
        column_names.extend(humid_variance_column_names)
        column_names.extend(humid_mean_column_names)
        column_names.extend(humid_min_column_names)
        column_names.extend(humid_max_column_names)
        column_names.extend(prec_variance_column_names)
        column_names.extend(prec_mean_column_names)
        column_names.extend(prec_min_column_names)
        column_names.extend(prec_max_column_names)

        # An empty but formatted DataFrame. This will be filled in the following.
        weatherfeatures_df = pd.DataFrame(columns=column_names)

        # Calculating the temperature features.
        frame_length = temperature_df.shape[1]
        interval_length = frame_length // number_of_temp_intervals

        for index in range(number_of_temp_intervals):

            if index < number_of_temp_intervals - 1:
                current_df = temperature_df[
                    temperature_df.columns[interval_length * index: interval_length * (index + 1)]]
            else:
                current_df = temperature_df[
                    temperature_df.columns[interval_length * (number_of_temp_intervals - 1): frame_length]]

            weatherfeatures_df['temp_variance-' + str(number_of_temp_intervals - index)] = current_df.var(axis=1)
            weatherfeatures_df['temp_mean-' + str(number_of_temp_intervals - index)] = current_df.mean(axis=1)
            weatherfeatures_df['temp_min-' + str(number_of_temp_intervals - index)] = current_df.min(axis=1)
            weatherfeatures_df['temp_max-' + str(number_of_temp_intervals - index)] = current_df.max(axis=1)

        # Calculating the humidity features.
        frame_length = humidity_df.shape[1]
        interval_length = frame_length // number_of_humid_intervals

        for index in range(number_of_humid_intervals):

            if index < number_of_humid_intervals - 1:
                current_df = humidity_df[humidity_df.columns[interval_length * index: interval_length * (index + 1)]]
            else:
                current_df = humidity_df[
                    humidity_df.columns[interval_length * (number_of_humid_intervals - 1): frame_length]]

            weatherfeatures_df['humid_variance-' + str(number_of_humid_intervals - index)] = current_df.var(axis=1)
            weatherfeatures_df['humid_mean-' + str(number_of_humid_intervals - index)] = current_df.mean(axis=1)
            weatherfeatures_df['humid_min-' + str(number_of_humid_intervals - index)] = current_df.min(axis=1)
            weatherfeatures_df['humid_max-' + str(number_of_humid_intervals - index)] = current_df.max(axis=1)

        # Calculating the precipitation features.
        frame_length = precipitation_df.shape[1]
        interval_length = frame_length // number_of_prec_intervals

        for index in range(number_of_prec_intervals):

            if index < number_of_prec_intervals - 1:
                current_df = precipitation_df[
                    precipitation_df.columns[interval_length * index: interval_length * (index + 1)]]
            else:
                current_df = precipitation_df[
                    precipitation_df.columns[interval_length * (number_of_prec_intervals - 1): frame_length]]

            weatherfeatures_df['prec_variance-' + str(number_of_prec_intervals - index)] = current_df.var(axis=1)
            weatherfeatures_df['prec_mean-' + str(number_of_prec_intervals - index)] = current_df.mean(axis=1)
            weatherfeatures_df['prec_min-' + str(number_of_prec_intervals - index)] = current_df.min(axis=1)
            weatherfeatures_df['prec_max-' + str(number_of_prec_intervals - index)] = current_df.max(axis=1)

        # Getting the non-weather columns.
        rawNonWeatherDF = self.rawXDataFrame[self.rawXDataFrame.columns[[not (
                bool(re.findall(r'temp_day', column_name)) or bool(re.findall(r'humid_day', column_name)) or bool(
            re.findall(r'prec_day', column_name))) for column_name in self.rawXDataFrame.columns]]]

        weekyear_state_df = rawNonWeatherDF[rawNonWeatherDF.columns[:2]]
        trend_influenza_df = rawNonWeatherDF[rawNonWeatherDF.columns[2:]]


        encoded_states_df = None
        encoded_weeks_df = None

        if encode_states:
            week_series = weekyear_state_df['state']
            label_encoder = LabelEncoder()
            label_encoder.fit(week_series)
            encoded_labels = label_encoder.transform(week_series)

            onehot_encoder = OneHotEncoder()
            onehot_encoder.fit(encoded_labels.reshape(-1, 1))
            onehot_labels = onehot_encoder.transform(encoded_labels.reshape(-1, 1))

            encoded_states_df = pd.DataFrame(onehot_labels.todense(),
                                             columns=weekyear_state_df['state'].unique())

        if encode_weeks:
            week_series = weekyear_state_df['year_week'].apply(lambda x: x[1])
            label_encoder = LabelEncoder()
            label_encoder.fit(week_series)
            encoded_labels = label_encoder.transform(week_series)

            onehot_encoder = OneHotEncoder()
            onehot_encoder.fit(encoded_labels.reshape(-1, 1))
            onehot_labels = onehot_encoder.transform(encoded_labels.reshape(-1, 1))

            encoded_weeks_df = pd.DataFrame(onehot_labels.todense(),
                                             columns=['week_' + str(i) for i in range(1, 54)])

        if encode_states and encode_weeks:
            return_df = pd.concat([weekyear_state_df, encoded_weeks_df, encoded_states_df, weatherfeatures_df, trend_influenza_df],
                                  axis=1)
        elif encode_states:
            return_df = pd.concat([weekyear_state_df, encoded_states_df, weatherfeatures_df, trend_influenza_df],
                                  axis=1)
        elif encode_weeks:
            return_df = pd.concat([weekyear_state_df, encoded_weeks_df, weatherfeatures_df, trend_influenza_df],
                                  axis=1)
        else:
            return_df = pd.concat([weekyear_state_df, weatherfeatures_df, trend_influenza_df], axis=1)

        return return_df


##################################
## Data Frame Manipulation Methods
##################################


# Unit test passed
def shift_append_target_weeks(X_df, y_series, shift, number_of_additional_weeks=0):
    """
    This function receives a X data frame and a y series. According to the number of additional weeks additional columns
    are appended to the y series. According to the shift parameter the y series or data frame is shifted upward. This
    basically implements shifting and adding target variables and modifying the feature matrix accordingly.

    :param X_df: A pandas.DataFrame, containing rows with names 'state', 'year_week'.
    :param y_series: A pandas.Series.
    :param shift: An int, determining by how many initial rows of the y_series and how many final rows of the
    X data frame are removed. These deletions implement an upward shift of the y_series.
    :param number_of_additional_weeks: An int, the number of additional columns added to y. Each additional column is
    shifted one index up with respect to the previous column.
    :return: A 2-tuple, the first entry is the modified X data frame and the second entry is the modified y.
    """
    shifted_X_df = None
    shifted_appended_y_series_df = None

    for current_state in X_df['state'].unique():

        # Indices of the current state refer to True the others refer to False
        state_indices = X_df['state'] == current_state

        # Looking at the columns associated with the current state.
        X_state_df = X_df[state_indices]
        # Shifting the dataframe by shift.
        X_state_df = X_state_df.iloc[: - (shift + number_of_additional_weeks)]

        # y is treated accordingly
        y_state_series = y_series[state_indices]

        current_container_y_state_series_df = None

        for current_week in range(number_of_additional_weeks + 1):

            lower_index = shift + current_week
            upper_index = - (number_of_additional_weeks - current_week)

            if upper_index == 0:
                current_y_state_series = y_state_series.iloc[lower_index:]
            else:
                current_y_state_series = y_state_series.iloc[lower_index:  upper_index]

            current_y_state_series = current_y_state_series.reset_index(drop=True)

            if current_container_y_state_series_df is not None:
                current_container_y_state_series_df = pd.concat(
                    [current_container_y_state_series_df, current_y_state_series], axis=1)
            else:
                current_container_y_state_series_df = current_y_state_series

        # constructing the new X and y
        if shifted_X_df is None:
            shifted_X_df = X_state_df
            shifted_appended_y_series_df = current_container_y_state_series_df
        else:
            shifted_X_df = pd.concat([shifted_X_df, X_state_df])
            shifted_appended_y_series_df = pd.concat(
                [shifted_appended_y_series_df, current_container_y_state_series_df])

        shifted_X_df = shifted_X_df.reset_index(drop=True)
        shifted_appended_y_series_df = shifted_appended_y_series_df.reset_index(drop=True)

    return shifted_X_df, shifted_appended_y_series_df


def get_wave_complement_interval_split(input_df, start_year, start_week, end_year, end_week):
    """
    This functions splits the input_df into two data frames according to the interval and
    its complement specified by the parameters. (This function can be used
    to perform a train - test split for instance.)

    :param input_df: A pandas.DataFrame, containing a row with the name 'year_week'
    :param start_year: An int, specifying the year the interval starts.
    :param start_week: An int, specifying the week the interval starts.
    :param end_year: An int, specifying the year the interval ends.
    :param end_week: An int, specifying the week the interval ends.
    :return: A 2-tuple, each entry of type pandas.DataFrame. The second data frame contains the rows of the input data
    frame associated with the time interval specified by the start year and week and end year and week. The first data
    frame contains the rows associated with the complement of the above time interval.
    """
    interval_bool_series = input_df['year_week'].apply(
        lambda x: (start_year <= x[0] and (start_week <= x[1] or start_year < x[0])) and (
                x[0] <= end_year and (x[1] <= end_week or x[0] < end_year)))

    interval_complement_bool_series = interval_bool_series.apply(
        lambda x: not x)

    interval_df = input_df[interval_bool_series]
    interval_complement_df = input_df[interval_complement_bool_series]

    return interval_complement_df, interval_df


# Unit test passed
def split_data(input_df, year_week_test_list):
    """
    This function splits the input data frame into two data frames according to the year_week_test list.
    That is to say the returned 2-tuple contains two data frames. The first containing the rows which have a year_week
    value not contained in the year_week_test_list and the second containing the rows which have a year_week value
    contained in the year_week_test_list.

    :param input_df: A pandas.DataFrame, containing a row with name 'year_week'
    :param year_week_test_list: A list, containing the year_week tuples for the selection.
    :return: A 2-tuple, containing two data frames.
    """
    test_bool_series = input_df['year_week'].apply(lambda x: x in year_week_test_list)
    train_bool_series = test_bool_series.apply(lambda x: not x)

    train_data_df = input_df[train_bool_series]
    test_data_df = input_df[test_bool_series]

    return train_data_df, test_data_df


def get_custom_cv_split_index_list(input_df, start_week=25, end_week=24, year_range_start=2005,
                                   year_range_end=2014, exclude_2009_bool=True):
    """
    This function returns a list of 2-tuples containing indices. The indices refer to the rows of the data frame which
    lie outside of the specified interval respectively inside the interval. The returned list can be used for instance
    for cross-validation and grid search.

    :param input_df: A pandas.DataFrame, containing a row with the name 'year_week'.
    :param start_week: An int, the start week of the interval.
    :param end_week: An int, the end week of the interval.
    :param year_range_start: An int, the start year of the interval.
    :param year_range_end: An int, the final year of the interval.
    :param exclude_2009_bool: A bool, specifying whether the year 2009 is a validation year or not.
    :return: A list, containing 2-tuples of indices. Each tuple contains the indices of the complement of the above
    specified interval and the indices of the interval. The indices refer to the rows of the input data frame.
    """
    cv_split_index_list = []

    year_range = range(year_range_start, year_range_end + 1)

    if exclude_2009_bool:
        year_range = [year for year in year_range if year != 2009]

    for year in year_range:
        # Splitting the data frame into a 9 year training period and a 1 year validation period.
        complement_df, interval_df = get_wave_complement_interval_split(input_df, year, start_week, year + 1,
                                                                        end_week)

        # Appending the numpy arrays indices to the returned list.
        cv_split_index_list.append((complement_df.index.values, interval_df.index.values))

    return cv_split_index_list


def remove_columns_with(input_df, column_str_list):
    """
    This function removes all columns containing at least one of the strs in the column_str_list.

    :param input_df: A pandas.DataFrame.
    :param column_str_list: A list, containing strs which should not be contained in any column name of the returned
    data frame.
    :return: A pandas.DataFrame, the specified columns are removed.
    """
    # Removing the columns with column names containing the a colunn string of column_strings.
    for column_string in column_str_list:
        input_df = input_df[
            input_df.columns[[not (bool(re.findall(column_string, column_name))) for column_name in input_df.columns]]]

    return input_df


def exclude_rows_by_states_2009_summer(weather_trend_influenza_df, y_df_series, valid_states_list=['all'], no_2009_bool=True,
                               only_seasonal_weeks_bool=False):
    """
    This function deletes rows from the input data frame and series defined by the other input parameters.

    :param weather_trend_influenza_df: A pandas.DataFrame, containing the features. The data frame should have the
    column names 'year_week', 'state'.
    :param y_df_series: A pandas.Series, containing the target values.
    :param valid_states_list: A list[str], the names of valid states.
    :param no_2009_bool: A bool, specifying whether the outlier year 2009 (swine flue epidemic) should be excluded.
    :param only_seasonal_weeks_bool: A bool, indicating whether only seasonal data is included. In other words whether
    rows associated to summer should be excluded.
    :return: A 2-tuple, a pandas.DataFrame and a pandas.Series containing only valid(as specified by the paremters)
    rows.
    """

    # Excluding specific states, years, periods of the year.
    # Excluding states that are not specified
    if valid_states_list[0] != 'all':
        states_indices = weather_trend_influenza_df['state'].apply(lambda x: x in valid_states_list)
        weather_trend_influenza_df = weather_trend_influenza_df[states_indices]
        y_df_series = y_df_series[states_indices]
        weather_trend_influenza_df = weather_trend_influenza_df.reset_index(drop=True)
        y_df_series = y_df_series.reset_index(drop=True)

    # Excluding 2009 if parameter is set accodringly
    if no_2009_bool:
        start_year = 2009
        start_week = 25
        end_year = 2010
        end_week = 24
        no_2009_indices = weather_trend_influenza_df['year_week'].apply(
            lambda x: not ((start_year <= x[0] and (start_week <= x[1] or start_year < x[0])) and (
                    x[0] <= end_year and (x[1] <= end_week or x[0] < end_year))))
        weather_trend_influenza_df = weather_trend_influenza_df[no_2009_indices]
        y_df_series = y_df_series[no_2009_indices]
        weather_trend_influenza_df = weather_trend_influenza_df.reset_index(drop=True)
        y_df_series = y_df_series.reset_index(drop=True)

    # Excluding hotter month if specified
    if only_seasonal_weeks_bool:
        seasonal_weeks_indices = weather_trend_influenza_df['year_week'].apply(lambda x: 47 < x[1] or x[1] < 20)
        weather_trend_influenza_df = weather_trend_influenza_df[seasonal_weeks_indices]
        y_df_series = y_df_series[seasonal_weeks_indices]
        weather_trend_influenza_df = weather_trend_influenza_df.reset_index(drop=True)
        y_df_series = y_df_series.reset_index(drop=True)

    return weather_trend_influenza_df, y_df_series


def get_specific_states(data_df, y_series_or_df, states_list):
    """
    This function returns a data frame and a corresponding series or data frame.
    These objects contain only the rows of the states specified in states_list.
    The index of the returned data frame and series are reset.

    :param data_df: A pandas.DataFrame, containing a row with name 'state'.
    :param y_series_or_df: A pandas.DataFrame or a pandas.Series.
    :param states_list: A list, containing the desired states.
    :return: A tuple, containing the a data frame and a series or data frame. Both contain only rows which have a
    'state' column value contained in the states_list.
    """

    state_indices = data_df['state'].apply(lambda x: x in states_list)
    return_df = data_df[state_indices]
    return_df = return_df.reset_index(drop=True)
    return_series = y_series_or_df[state_indices]
    return_series = return_series.reset_index(drop=True)

    return return_df, return_series