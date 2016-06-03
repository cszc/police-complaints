import pandas as pd
import psycopg2
import sys

def go(output_fn):
    '''Generate dataframe with features from database'''

    conn = psycopg2.connect("dbname = police user = lauren password = llc")

    #Queries for features
    outcome = 'SELECT crid, officer_id, "Findings Sustained" FROM dependent_dum;'

    alleg = "SELECT crid, a.officer_id, tractce10, o.race_edit As race, \
                (CASE WHEN EXTRACT(dow FROM a.dateobj) NOT IN (0, 6) THEN 1 ELSE 0 END) AS weekend, \
                (CASE WHEN o.rank IS NOT NULL THEN o.rank ELSE 'UNKNOWN' END) AS rank, \
                (CASE WHEN investigator_name IN (SELECT concat_ws(', ', officer_last, officer_first) \
                FROM officers) THEN 1 ELSE 0 END) AS police_investigator \
                FROM allegations as a LEFT JOIN officers as o \
                ON (a.officer_id = o.officer_id) \
                LEFT JOIN investigators AS i ON (a.investigator_id = i.investigator_id) \
                WHERE tractce10 IS NOT NULL;"

    invest1 = "SELECT * FROM investigator_beat_dum1;"

    invest2 = "SELECT * FROM investigator_beat_dum2;"

    age = "SELECT crid, officer_id, officers_age, (officers_age^2) AS agesqrd FROM ages;"

    data311 = "SELECT * FROM time_distance_311;"

    datacrime = "SELECT * FROM time_distance_crime;"

    priors = "SELECT * FROM prior_complaints;"

    acs = "SELECT tract_1, pct017, pct1824, pct2534, pct3544, pct4554, pct5564, pct6500, \
                ptnla, ptnlb, ptnlwh, ptnloth, ptl, ptlths, pthsged, ptsomeco, ptbaplus, ptpov, pctfb \
                FROM acs;"

    officer_gender = "SELECT officer_id, gender FROM officers;"

    outcome_df = pd.read_sql(outcome, conn)
    alleg_df = pd.read_sql(alleg, conn)
    invest1_df = pd.read_sql(invest1, conn)
    invest2_df = pd.read_sql(invest2, conn)
    age_df = pd.read_sql(age, conn)
    data311_df = pd.read_sql(data311, conn)
    datacrime_df = pd.read_sql(datacrime, conn)
    priors_df = pd.read_sql(priors, conn)
    acs_df = pd.read_sql(acs, conn)
    off_gender_df = pd.read_sql(officer_gender, conn)
    #Close connection to database after queries

    conn.commit()
    conn.close()

    data311_df.drop_duplicates('crid', inplace = True)
    datacrime_df.drop_duplicates('crid', inplace = True)
    acs_df.drop_duplicates(inplace = True)

    #Merge (join) dataframes on shared keys
    df_final = outcome_df.merge(alleg_df, on = ['crid', 'officer_id'], how = 'left')\
                .merge(invest1_df.drop('index', axis = 1), on = ['crid', 'officer_id'], how = 'left')\
                .merge(invest2_df.drop('index', axis = 1), on = ['crid', 'officer_id'], how = 'left')\
                .merge(age_df, on = ['crid', 'officer_id'], how = 'left')\
                .merge(data311_df, on = 'crid', how = 'left').merge(datacrime_df, on = 'crid', how = 'left')\
                .merge(priors_df, on = ['crid', 'officer_id'], how = 'left')\
                .merge(acs_df, how = 'left', left_on = 'tractce10', right_on = 'tract_1')\
                .merge(off_gender_df, how = 'left', on = 'officer_id')

    #Dummies for race and rank and drop unneeded columns
    race_dummies = pd.get_dummies(df_final['race'], prefix = 'Race', prefix_sep = ' ', dummy_na = True)
    rank_dummies = pd.get_dummies(df_final['rank'], prefix = 'Rank', prefix_sep = ' ', dummy_na = True)
    gender_dummies = pd.get_dummies(df_final['gender'], prefix = 'Gender', prefix_sep = ' ', dummy_na = True)
    df_final = pd.concat([df_final, race_dummies, rank_dummies, gender_dummies], axis = 1)
    df_final.drop(['tract_1', 'tractce10', 'race', 'rank', 'gender'], axis = 1, inplace = True)

    df_final.to_csv(output_fn)

def impute_gender(df, name_col, gender_col):
    '''Fills in missing gender using Genderize.io API.
    Takes the dataframe, column with first name, and column with gender'''
    for i, row in df.iterrows():
        if pd.isnull(df.ix[i , gender_col]):
            name = df.ix[i , name_col]
            result = requests.get('https://api.genderize.io/?name=' + name)
            gender = result.json()['gender'][:1]
            #Capitalize gender to match the rest of the table
            df.set_value(index = i, col = gender_col, value = gender.title())
    return df


if __name__ == '__main__':
    go(sys.argv[1])