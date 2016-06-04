"""
The explore module (1/5) reads in and explores data. Provides functions for
generating summary statistics and histograms.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import time, datetime, re, pickle, sys
import seaborn as sns
sns.set(color_codes=True)


def read_in(path, filetype = 'csv'):
    """
    Read file to pandas dataframe. Currently only csv is allowed and
    is the default. Plan on extending later.
    """
    if filetype == 'csv':
        return pd.read_csv(path, index_col=0)
    else:
        sys.exit("Filetype not supported.")


def get_null_freq(df):
    """
    For a given DataFrame, calculates how many values for
    each variable is null.

    Returns a table (dataframe) of columns and null values.

    Taken From: https://github.com/yhat/DataGotham2013/blob/master/notebooks/3%20-%20Importing%20Data.ipynb
    """
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    return pd.crosstab(df_lng.variable, null_variables)


def camel_to_snake(column_name):
    """
    Converts a string that is camelCase into snake_case.
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    Taken From: https://github.com/yhat/DataGotham2013/blob/master/notebooks/3%20-%20Importing%20Data.ipynb
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_columns(df):
    '''
    Converts column names in dataframe to snake_case.
    Returns dataframe with modified column names.
    '''
    df.columns = [camel_to_snake(col) for col in df.columns]
    return df


def make_hist(df, title, num_bins = 8, code_percentile=.99):
    '''
    Takes data, title, number of bins (max 10), and percentile.
    Outputs a histogram.
    '''
    title = title.title()
    data_list = df.dropna().tolist()
    top_code_val = df.quantile(code_percentile)
    distinct_vals = len(set(data_list))
    num_bins = min(distinct_vals, num_bins)

    plt.style.use('ggplot')
    df.hist(bins = np.linspace(0, top_code_val, num_bins + 1), normed=True)
    plt.xlabel(title)
    plt.title('Histogram of ' + title.replace("_", " "))
    plt.tight_layout()
    plt.savefig('Histogram_' + title + '.png', format='png')
    plt.close()


def make_bar():
    df = pd.DataFrame(data={"age": randint(10, 50, 1000),
                        "response": [choice(['Yes', 'No']) for i in range(1000)]})

    df['age_group'] = pd.cut(df.age, bins=[g for g in range(10, 51, 5)], include_lowest=True)
    df.head()

    sns.countplot(y='response', hue='age_group', data=df, palette="Greens_d")

def summarize_data(df, html = True):
    '''
    Creates summary statistics and histograms.
    Outputs histograms as png.
    If filename given, exports tables to html.
    Otherwise, writes the tables to CSVs.
    '''

    stats = ('summary_statistics', df.describe())
    nulls = ('null_counts', get_null_freq(df))
    modes = ('mode_values', df.mode())
    summaries = [stats, nulls, modes]

    #Export Tables to html or csv
    for name, summary_df in summaries:
        row, col = summary_df.shape
        if col > row:
            summary_df = summary_df.T
        if html:
            filename = str(name + '.html')
            html_str = summary_df.to_html(open(filename, 'w+'))
        else:
            filename = str(name + '.csv')
            summary_df.to_csv(open(filename, 'w+'))

    #Make Histograms of Numeric Column Types
    numeric_df = df.select_dtypes(include=[np.number])
    names = numeric_df.columns.values
    for name in names:
        column_data = df[name]
        column_title = name
        make_hist(column_data, column_title)

    #Make Histograms of Numeric Column Types
    nonnumeric_df = df.select_dtypes(include=[object])
    nonnumeric_names = nonnumeric_df.columns.values
    print('Not visualized: {}'.format(nonnumeric_names))

    # for name in non_numeric_names:
    #
    #     title = name.title()
    #     data_list = df[name].dropna().tolist()
    #
    #     plt.style.use('ggplot')
    #     plt.xlabel(title)
    #     plt.title('Histogram of ' + title.replace("_", " "))
    #     plt.tight_layout()
    #     plt.savefig('Histogram_' + title + '.png', format='png')
    #     plt.close()
    #     x = ['a','b','c','d']
    #     y = [1,2,3,4]
    #     plt.plot(y)
    #     plt.xticks(range(len(x)), x)
    #     plt.show()


def go(path):
    '''
    all-in-one function that reads a path into a pandas dataframe,
    snakes the columns, and summarizes the data.
    '''
    df = read_in(path)
    df = snake_columns(df)
    #create data dict
    df.dtypes.to_csv('{0}_data-dict_{1}'.format(TIMESTAMP, path.strip(".csv")), header=True)
    summarize_data(df)
    file_name = "{0}_cleaned-df_{1}.p".format(TIMESTAMP, path.strip(".csv"))
    data = [df] #need to fill in with things that we want to pickle
    pickle.dump( data, open(file_name, "wb" ) )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('csv')
    args = parser.parse_args()

    TS = time.time()
    TIMESTAMP = datetime.datetime.fromtimestamp(TS).strftime('%Y-%m-%d')

    go(args.csv)
