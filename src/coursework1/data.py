# Data preparation and understanding code
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import pathlib
import time
import math
from datetime import datetime,timedelta
warnings.simplefilter(action='ignore', category=FutureWarning)
def print_general_statistics(df):
    """ print the general information about the dataframe;
        print first 5 rows and all the columns of the data frame
        demonstrate number of row and column of the data frame
        print the data types; general statics information of the data frame
    Args:
        df: The data frame imported.
    """
    pd.set_option('display.max_columns', None)# set all the columns visible in the terminal printing
    pd.set_option('display.width', None)
    print("\nthe first 5 rows of dataframe :\n")
    print(df.head(12))
    print("\nThe Rows and Columns number:\n")
    print("\nRow Number :" + str(df.shape[0]))
    print("\nColumn Number :"+str(df.shape[1]))
    print("\nColumn data types:\n")
    print(df.dtypes)
    print("\nStatistics:\n")
    print(df.describe())  # Add your code inside the brackets
def null_data_disposal(df):
    """  if the dataset has null values, the function will delete the row in dataframe with
    null value. The function will return the dataframe without null.

        Args:
            df: The dataframe prepared to delete null values
        Return:
            df: The dataframe after deleting null values
    """
    result=df.isnull().any().any()
    if result == True:
        print('the dataframe has null value')
        print('the null row will be deleted\n null data preprocessing finish')
        df =df.dropna()
        return (df)
    else:
        print('the dataframe has no null value\n null data preprocessing finish')
        return (df)
def time_stamp_format_convert(df):
    """  the function is used to convert the ' timestamp'   to datetype data
    also, it converts all the timestamp to (Minutes:seconds.microseconds) format.
         Args: df:the dataframe used to disposal
               invalid_times: a new dataframe with invalid timestamp column
         Return:df_return: the dataframe after timestamp format convert from string to datetyoe.
    """
    print(df.loc[20928,'timestamp'])
    df.loc[20928,'timestamp']='05:48.0'
    # time data "6/25/2022 14:05" doesn't match format "M:%s%f". at position 20928. I manually change the format of time
    # based on continuously distribution of time
    df['timestamp_datetype'] = pd.to_datetime(df['timestamp'], format='%M:%S.%f', errors='coerce')
    invalid_times = df[df['timestamp_datetype'].isna()]
    if invalid_times.empty:
        print('all the timestamp_datetype are valid ')
    else:
         print('\nsome of timestamp_datetype are invalid\n')
         print(' here are the invalid formats')
         print(invalid_times)
    print(df.loc[20928, 'timestamp_datetype'])
    df_return=df
    return(df_return)
def breaking_point_detection(df):
    ''' detection the breaking point to break the continuity of time_stamp
        give an output of list of index with breaking points
        Args: df : input data frame
        Return: list_of_breaking_points: the list of the index of  breaking point
    '''
    list_of_difference=[]
    breaking_point_index=[]
    for i in range(df.shape[0]):
        if i !=(df.shape[0]-1):
            difference=df.loc[(i+1),'timestamp_datetype']-df.loc[i,'timestamp_datetype']
            if difference.total_seconds()==(1/10):
                pass
            else:
                  breaking_point_index.append(i+1)
                  list_of_difference.append(difference.total_seconds())

    print(list_of_difference)
    print(breaking_point_index)
    return (breaking_point_index,list_of_difference)
def interpolation(breaking_point_index,list_of_difference,df):
    '''1
        some data points are not continuous and they are not breaking points on timestamp.
        the data of sensor is delayed sometimes.
        so a function of interpolation is applied to detect whether difference of timestamp between 2 points
        is 0.2 second（normally one second) In this case, I interpolate a new piece of data with 0.1s difference
        between 2 points.

        Args:
        breaking_point_index (list): indices where breaking points locate
        list_of_difference (list): time differences between consecutive points
        df (pd.DataFrame): input dataframe with discontinuously timeseries

        Returns:
        pd.DataFrame: Dataframe with interpolated rows added where needed.
    '''
    # create the number of points implementing interpolation
    number_of_interpolation_point=0
    list_of_interpolation_index=[]
    for i in range(len(list_of_difference)):

        if list_of_difference[i] == 0.2:
            df1 = df.iloc[:(number_of_interpolation_point+breaking_point_index[i]),:]
            df2 = df.iloc[(number_of_interpolation_point+breaking_point_index[i]):,:]
            # create a new row to make time_stamp continuously. And fill with nan for following interpolation
            new_row = pd.DataFrame({'accX': [math.nan], 'accY':[math.nan], 'accZ':[math.nan], 'gyroX':[math.nan],
            'gyroY':[math.nan], 'gyroZ':[math.nan],'timestamp':[math.nan],'Activity':[math.nan],
            'timestamp_datetype':[df.loc[breaking_point_index[i]+number_of_interpolation_point,'timestamp_datetype']
                                  -timedelta(seconds=0.1)]})
            new_row['timestamp'] = new_row['timestamp_datetype'].dt.strftime('%M:%S.%f')
            df = pd.concat([df1, new_row, df2]).reset_index(drop=True)
            number_of_interpolation_point += 1
            list_of_interpolation_index.append(breaking_point_index[i]+number_of_interpolation_point)

        elif ( list_of_difference[i] == 0.4):
            print(breaking_point_index[i])
            df1 = df.iloc[:(number_of_interpolation_point+breaking_point_index[i]),:]
            df2 = df.iloc[(number_of_interpolation_point+breaking_point_index[i]):,:]
            number_of_new_rows = 4
            for j in range(number_of_new_rows):
                new_row = pd.DataFrame({'accX': [math.nan], 'accY': [math.nan], 'accZ': [math.nan], 'gyroX': [math.nan],
                'gyroY': [math.nan], 'gyroZ': [math.nan], 'timestamp': [math.nan],'Activity': [math.nan],
                'timestamp_datetype': [df.loc[breaking_point_index[i-1] + number_of_interpolation_point,
                'timestamp_datetype'] + (j+1)*timedelta(seconds=0.1)]})
                new_row['timestamp'] = new_row['timestamp_datetype'].dt.strftime('%M:%S.%f')
                df1=df1.append(new_row,ignore_index=True)
                number_of_interpolation_point += 1
                list_of_interpolation_index.append(breaking_point_index[i] + number_of_interpolation_point)
                df = pd.concat([df1,df2]).reset_index(drop=True)
    for col in ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ','Activity']:
        df[col]=df[col].interpolate(method='pchip')
    print('the number of data points interpolated :'+str(number_of_interpolation_point))
    print('the index of data points added')
    print(list_of_interpolation_index)
    df.reset_index(drop=True, inplace=True)
    return df


def timestamp_delete(breaking_point_index,list_of_difference,df):
    number_of_delete_point = 0
    list_of_index=[]
    for i in range(len(list_of_difference)):
        if (list_of_difference[i]<0.1 and list_of_difference[i] >=-0.1):
            list_of_index.append(breaking_point_index[i])
            df=df.drop(index=breaking_point_index[i])
            number_of_delete_point+=1
    print('the number of repetitive data points deleted :' + str(number_of_delete_point))
    print('the index of data points deleted')
    print(list_of_index)
    df.reset_index(drop=True, inplace=True)
    return(df)
 #def dataframe_list_creator():
    list_of_dataframe=[]

# 时间要是连续的 activities 要是连续的（1-0）创建 different frame
def outlier_disposal(df):
    window_size = 5
    threshold = 3
    outliers = {}
    for column in df.columns:
        if column not in ['Activity', 'timestamp', 'timestamp_datetype']:
            rolling_mean = df[column].rolling(window=window_size, min_periods=0).mean()
            rolling_standard_deviation = df[column].rolling(window=window_size, min_periods=0).std()
            upper_bound = df[column] < (rolling_mean - threshold * rolling_standard_deviation)
            lower_bound = df[column] > (rolling_mean + threshold * rolling_standard_deviation)
            outlier_condition = (upper_bound | lower_bound)
            outliers[column] = df[outlier_condition]

    print(outliers)
def different_activity_frame_division(df):
     df_0 = df[df['Activity'] == 0]
     df_1 = df[df['Activity'] == 1]
     return df_0, df_1

def statics_hisdiagram(df):
    plt.figure()

    for column in df:
        if column not in ['Activity', 'timestamp', 'timestamp_datetype']:
            df[column].hist(bins=15, width=2)
            plt.title(column)
            if column in ['accX','accY','accZ']:
                 plt.xlabel('m/s^(-2)')
            else:
                plt.xlabel('')
            plt.ylabel('Frequency')
            plt.show()

def statistics_boxplot(df):



if __name__ == '__main__':

    df_raw = pd.read_csv('dataset.csv')
    print_general_statistics(df_raw)
    print_general_statistics(df_raw)
    df_after_null_preprocess = null_data_disposal(df_raw)
    df_after_time_stamp_convert = time_stamp_format_convert(df_after_null_preprocess)
    print_general_statistics(df_after_time_stamp_convert)
    print(df_after_time_stamp_convert.loc[20928, 'timestamp'])
    list_of_breaking_points,list_of_difference=breaking_point_detection(df_after_time_stamp_convert)
    df_after_delete=timestamp_delete(list_of_breaking_points,list_of_difference,df_after_time_stamp_convert)
    list_of_breaking_points, list_of_difference = breaking_point_detection(df_after_delete)
    df_interpolation=interpolation(list_of_breaking_points,list_of_difference,df_after_delete)
    list_of_breaking_points,list_of_difference=breaking_point_detection(df_interpolation)
    df_interpolation.to_csv('output_file.csv', index=True)
    df_activity_0,df_activity_1= different_activity_frame_division(df_interpolation)
    outlier_disposal(df_activity_0)
    outlier_disposal(df_activity_1)
    statics_hisdiagram(df_activity_0)
    statics_hisdiagram(df_activity_1)
