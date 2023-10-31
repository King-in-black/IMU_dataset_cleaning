# Data preparation and understanding code
import pandas as pd
import pathlib
import time
from datetime import datetime
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
    print(df.head(5))
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
    list_of_breaking_points=[]
    for i in range(df.shape[0]):
        if i !=(df.shape[0]-1):
            difference=df.loc[(i+1),'timestamp_datetype']-df.loc[i,'timestamp_datetype']
            if difference.total_seconds()==(1/10):
                pass
            else:list_of_breaking_points.append(i+1)
    print(list_of_breaking_points)
    return (list_of_breaking_points)

if __name__ == '__main__':
    # Use Pathlib.Path to read a file using the location relative to this file
    raw_data_file = pathlib.Path(__file__).parent.joinpath('accelerometer+gyro+mobile+phone+dataset', 'accelerometer_gyro_mobile_phone_dataset.csv')
    # Call the create_dataframe function, passing the csv file as argument
    df_raw = pd.read_csv(raw_data_file)
    print_general_statistics(df_raw)
    df_after_null_preprocess=null_data_disposal(df_raw)
    df_after_time_stamp_convert=time_stamp_format_convert(df_after_null_preprocess)
    print_general_statistics(df_after_time_stamp_convert)
    print(df_after_time_stamp_convert.loc[20928, 'timestamp'])
    new_dataframe_created_according_to_timestamp_contineity(df_after_time_stamp_convert)
