# Data preparation and understanding code
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import math
from datetime import timedelta
import os

warnings.simplefilter(action="ignore", category=FutureWarning)


# ignore the waring from df.approve
def print_general_statistics(df):
    """
        print the general information about the dataframe;
        print first 5 rows and all the columns of the data frame;
        demonstrate number of row and column of the data frame;
        print the data types; general statics information of the data frame
    Args:
        df: The data frame imported.
    """
    pd.set_option(
        "display.max_columns", None
    )  # set all the columns visible in the terminal printing
    pd.set_option("display.width", None)
    print("\nthe first 5 rows of dataframe :\n")
    print(df.head(12))
    print("\nThe Rows and Columns number:\n")
    print("\nRow Number :" + str(df.shape[0]))
    print("\nColumn Number :" + str(df.shape[1]))
    print("\nColumn data types:\n")
    print(df.dtypes)
    print("\nStatistics:\n")
    print(df.describe())  # Add your code inside the brackets


def null_data_disposal(df):
    """if the dataset has null values, the function will delete the row in dataframe with
    null value. The function will return the dataframe without null.

        Args:
            df: The dataframe prepared to delete null values
        Return:
            df: The dataframe after deleting null values
    """
    result = df.isnull().any().any()
    # return the boolean value by detecting whether there is a null value
    if result == True:
        print("the dataframe has null value")
        print("the null row will be interpolated later\n null data preprocessing finish")
        return df
    else:
        print("the dataframe has no null value\n null data preprocessing finish")
        return df


def time_stamp_format_convert(df):
    """
    the function is used to convert the ' timestamp'   to datetype data
    also, it converts all the timestamp to (Minutes:seconds.microseconds) format.
         Args: df:the dataframe used to disposal
               invalid_times: a new dataframe with invalid timestamp column
         Return:df_return: the dataframe after timestamp format convert from string to datetype.

    """
    print("one of the dateformat is wrong, just correct directly without a function")
    print(df.loc[20928, "timestamp"])
    df.loc[20928, "timestamp"] = "05:48.0"
    # time data "6/25/2022 14:05" doesn't match format "M:%s%f". at position 20928. I manually change the format of time
    # based on continuously distribution of time
    df["timestamp_datetype"] = pd.to_datetime(
        df["timestamp"], format="%M:%S.%f", errors="coerce"
    )
    # add a column in the dataframe by converting string to datetype data
    invalid_times = df[df["timestamp_datetype"].isna()]
    # return to a Series of date whether the timestamp has invalid format , which could not change to %M:%S.%f"
    if invalid_times.empty:
        print("all the timestamp_datetype are valid ")
    else:
        print("\nsome of timestamp_datetype are invalid\n")
        print(" here are the invalid formats")
        print(invalid_times)
    print(df.loc[20928, "timestamp_datetype"])
    df_return = df
    return df_return


def breaking_point_detection(df):
    """
    The continuity of time series should be ensured for following analysis
    detect the breaking point to break the continuity of time_stamp
    give an output of list of index with breaking points
    also calculate the difference in the time length of breaking points

         Args:     df : input dataframe
         Return:   breaking_point_index: the list of the index of breaking points locating in dataframe
                   list_of_difference: the list of time difference between breaking points

    """
    list_of_difference = []
    breaking_point_index = []
    for i in range(df.shape[0]):
        if i != (df.shape[0] - 1):
            difference = (
                df.loc[(i + 1), "timestamp_datetype"] - df.loc[i, "timestamp_datetype"]
            )
            # calculate the time delta, the normal difference between 2 time points is 0.1 second because the frequency
            # of collection data of sensor is 10Hz
            if difference.total_seconds() == (1 / 10):
                pass
            else:
                breaking_point_index.append(i + 1)
            # append the breaking points into the list.
                list_of_difference.append(difference.total_seconds())

    print(list_of_difference)
    print(breaking_point_index)
    return (breaking_point_index, list_of_difference)

def interpolation(df):
    """
    some data points are not continuous and they are not typical breaking points on timestamp.
    some breaking points are caused by delay of signal.
    so a function of interpolation is applied to detect whether difference of timestamp between 2 points
    is 0.2 second (normally 0.1s) In this case, I interpolate a new piece of data with 0.1s difference
    between 2 points.

    Args:
    df (pd.DataFrame): input dataframe with discontinuously timeseries

    Returns:
    pd.DataFrame: Dataframe with interpolated rows added where needed.
    """
    breaking_point_index, list_of_difference = breaking_point_detection(df)
    list_of_interpolation_index = [0]
    list_of_dataframe = []
   #  Take the index into a list with 0.2s time difference
    for i in range(len(list_of_difference)):
        if list_of_difference[i] == 0.2:
            list_of_interpolation_index.append(breaking_point_index[i])
    list_of_interpolation_index.append(df.shape[0])
    # Construct a list of dataframes with interpolated rows.
    for i in range(len(list_of_interpolation_index) - 1):
        start = list_of_interpolation_index[i]
        end = list_of_interpolation_index[i + 1]
        list_of_dataframe.append(df.loc[start:end - 1])
        if end != df.shape[0]:  # Avoid adding a new row after the last point
            new_row = pd.DataFrame({
                "accX": [math.nan], "accY": [math.nan], "accZ": [math.nan],
                "gyroX": [math.nan], "gyroY": [math.nan], "gyroZ": [math.nan],
                "timestamp": [math.nan], "Activity": [math.nan],
                "timestamp_datetype": [df.loc[end, "timestamp_datetype"] - timedelta(seconds=0.1)]
            })
            # append dataframe into a list
            list_of_dataframe.append(new_row)
    # Combine all the dataframe and new rows into a new dataframe
    new_df = pd.concat(list_of_dataframe).reset_index(drop=True)
    # Interpolate the data points with piecewise cubic hermite interpolating polynomial method
    for col in ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "Activity"]:
        new_df[col] = new_df[col].interpolate(method="pchip")
    return new_df


def timestamp_delete(df):
    '''
    The function is used to delete the repetitive timestamp and inverse  data in the time series in the dataframe
    Args:
        df: the dataframe prepare to delete the repetitive timestamp

    Returns:
        df_new: the dataframe after deletion of timestamp

    '''
    breaking_point_index, list_of_difference = breaking_point_detection(df)
    number_of_delete_point = 0
    list_of_index = []
    # drop the row in  the same or wrong order of time seires
    for i in range(len(list_of_difference)):
        if list_of_difference[i] < 0.1 and list_of_difference[i] >= -0.1:
            list_of_index.append(breaking_point_index[i])
            df = df.drop(index=breaking_point_index[i])
            number_of_delete_point += 1
    print(
        "the number of repetitive data points deleted :" + str(number_of_delete_point)
    )
    print("the index of data points deleted")
    print(list_of_index)
    df.reset_index(drop=True, inplace=True)
    return df


def outlier_disposal(df):
    '''
    the function is used rolling methods with a window to detect the outlier. Because the data has a tendency
    to increase or decrease, the static method to indicate outliers is pretty unsuitable.

    Args:
        df: dataframe used to detect outliers into dataframe

    Returns:
        outliers: a dataframe with outliers in the dataframe

    '''
    # set window_size and threshold
    window_size = 5
    threshold = 3
    outliers = {}
    for column in df.columns:
        if column not in ["Activity", "timestamp", "timestamp_datetype"]:
            rolling_mean = df[column].rolling(window=window_size, min_periods=0).mean()
            rolling_standard_deviation = (
                df[column].rolling(window=window_size, min_periods=0).std()
            )
            # if a data point is higher or lower a mean with 3 s.d , it will be recognised as an outlier
            upper_bound = df[column] < (
                rolling_mean - threshold * rolling_standard_deviation
            )
            lower_bound = df[column] > (
                rolling_mean + threshold * rolling_standard_deviation
            )
            outlier_condition = upper_bound | lower_bound
            outliers[column] = df[outlier_condition]

    if outliers.empty():
        print( 'there is no outliers')
    print(outliers)
    return(outliers)


def different_activity_frame_division(df):
    '''
    the function is used to divide dataframe according to number of Activity in the row of data.
    Activity =0 is for one dataframe, and Activity =1 is for another dataframe.

    Args:
        df: the input dataframe not classified by Activity

    Returns:
        df_0:the input dataframe  classified by Activity =0
        df_1:the input dataframe  classified by Activity =1
    '''
    df_0 = df[df["Activity"] == 0]
    df_1 = df[df["Activity"] == 1]
    return df_0, df_1

def statics_histgram(df):
    '''

    Args:
        df:

    Returns:

    '''
    plt.figure()
    for column in df:
        if column not in ["Activity", "timestamp", "timestamp_datetype"]:
            df[column].hist(bins=15, width=2)
            plt.title(column)
            if column in ["accX", "accY", "accZ"]:
                plt.xlabel("m/s^(-2)")
            else:
                plt.xlabel("radian per second")
            plt.ylabel("Frequency")
            plt.show()


def statistics_boxplot(df):
    df.boxplot(column=["accX", "accY", "accZ", "gyroX", "gyroY", "gyroY"])
    plt.show()


def smoothing(df):
    for column in df:
        if column not in ["Activity", "timestamp", "timestamp_datetype"]:
            df.loc[:, column] = df[column].ewm(alpha=0.7).mean()
    return df


def smoothing_all(df):
    list_of_breaking_points, list_of_difference = breaking_point_detection(df)
    list_of_breaking_points.append(df.shape[0])
    list_of_breaking_points.sort()
    list_of_dataframe = []
    start = 0
    for breaking_point in list_of_breaking_points:
        if breaking_point != df.shape[0]:
            new_df = df.loc[start : (breaking_point - 1), :]
            new_df = smoothing(new_df)
            list_of_dataframe.append(new_df)
            start = breaking_point
    df_after_smoothing = pd.concat(list_of_dataframe)
    return df_after_smoothing


def data_preprocessing():

    df_raw = pd.read_csv("dataset.csv")
    df_after_null_preprocess = null_data_disposal(df_raw)
    print_general_statistics(df_raw)
    print_general_statistics(df_raw)
    df_after_null_preprocess = null_data_disposal(df_raw)
    df_after_time_stamp_convert = time_stamp_format_convert(df_after_null_preprocess)
    print_general_statistics(df_after_time_stamp_convert)
    print(df_after_time_stamp_convert.loc[20928, "timestamp"])
    list_of_breaking_points, list_of_difference = breaking_point_detection(
        df_after_time_stamp_convert
    )
    df_after_delete = timestamp_delete(
        list_of_breaking_points, list_of_difference, df_after_time_stamp_convert
    )
    list_of_breaking_points, list_of_difference = breaking_point_detection(
        df_after_delete
    )
    df_interpolation = interpolation_2(df_after_delete)

    df_activity_0, df_activity_1 = different_activity_frame_division(df_interpolation)
    outlier_disposal(df_activity_0)
    outlier_disposal(df_activity_1)
    statics_histgram(df_activity_0)
    statics_histgram(df_activity_1)
    statistics_boxplot(df_activity_0)
    statistics_boxplot(df_activity_1)
    a,b=breaking_point_detection(df_interpolation)
    df_interpolation.to_csv("output_file.csv", index=True)

data_preprocessing()