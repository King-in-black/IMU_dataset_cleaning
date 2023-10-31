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
