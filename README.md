# police-complaints

Project for Machine Learning for Public Policy, Spring 2016

A project to evaluate the predictive power of complainant demographic data for predicting the outcome of police misconduct complaints in Chicago.

## Data
The original dataset was obtained through the Freedom of Information Act (FOIA) by the [Invisible Institute](http://invisible.institute/), an investigative journalism nonprofit based in Chicago. The dataset contains 56,000 misconduct complaint records for approximately 8,500 Chicago police officers since 2011. In total, we dropped 37,689 observations from the original dataset, decreasing the total number of observations in our base dataset from 56,384 to 18,695. We kept observations that fell within March 2011 - December 2014 as those are the most reliable. We added 311 and crime data from the City of Chicago open data portal, ACS data, and Tiger Census Shapefiles to generate additional features.

## Data Pipeline

- Use scripts in db_tools to upload data from CSV to PostGreSQL database
- Use scripts in features and geocoding to generate and save features
- Use scripts in final_data to export joined tables to CSV for use with SKLearn
- Run experiments using pipeline.py
- Evaluate results, output metrics, and generate plots using results

## Dependencies

- Python 3.4
- argparse
- pickle
- json
- itertools
- csv
- matplotlib
- pandas
- numpy
- scipy
- pydoc
- [UnbalancedDataset](https://github.com/fmfn/UnbalancedDataset)

## Credit
Thanks to the [DSSG Cincinnati team](https://github.com/dssg/cincinnati) for providing some useful code snippets for our pipeline and to the Invisible Institute for providing additional background and context for the data
