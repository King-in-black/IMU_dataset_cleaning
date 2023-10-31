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



if __name__ == '__main__':
    df_raw = pd.read_csv(dataset.csv)
    print_general_statistics(df_raw)
    print_general_statistics(df_raw)
    df_after_null_preprocess = null_data_disposal(df_raw)
    df_after_time_stamp_convert = time_stamp_format_convert(df_after_null_preprocess)
    print_general_statistics(df_after_time_stamp_convert)
    print(df_after_time_stamp_convert.loc[20928, 'timestamp'])
    print(breaking_point_detection(df_after_time_stamp_convert))