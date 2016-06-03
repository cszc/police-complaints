import pandas as pd
import psycopg2

def go():
    '''Generate dataframe with features from database'''

    conn = psycopg2.connect("dbname = police user = lauren password = llc")

    #Queries for features
    alleg = "SELECT crid, officer_id, tractce10,\
                (CASE WHEN EXTRACT(dow FROM dateobj) NOT IN (0, 6) THEN 1 ELSE 0 END) AS weekend \
                FROM allegations \
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

    alleg_df = pd.read_sql(alleg, conn)
    invest1_df = pd.read_sql(invest1, conn)
    invest2_df = pd.read_sql(invest2, conn)
    age_df = pd.read_sql(age, conn)
    data311_df = pd.read_sql(data311, conn)
    datacrime_df = pd.read_sql(datacrime, conn)
    priors_df = pd.read_sql(priors, conn)
    acs_df = pd.read_sql(acs, conn)

    #Close connection to database after queries
    conn.commit()
    conn.close()

    data311_df.drop_duplicates('crid', inplace = True)
    datacrime_df.drop_duplicates('crid', inplace = True)
    acs_df.drop_duplicates(inplace = True)

    #Merge (join) dataframes on shared keys
    df_final = alleg_df.merge(invest1_df.drop('index', axis = 1), on = ['crid', 'officer_id'], how = 'left')\
                .merge(invest2_df.drop('index', axis = 1), on = ['crid', 'officer_id'], how = 'left')\
                .merge(age_df, on = ['crid', 'officer_id'], how = 'left')\
                .merge(data311_df, on = 'crid', how = 'left').merge(datacrime_df, on = 'crid', how = 'left')\
                .merge(priors_df, on = ['crid', 'officer_id'], how = 'left')\
                .merge(acs_df, how = 'left', left_on = 'tractce10', right_on = 'tract_1')

    #Drop sequential index column
    #df_final.drop('index', axis = 1, inplace = True)
    df_final.drop(['tract_1', 'tractce10'], inplace = True)

    df_final.to_csv("queriedFeatureResults.csv")

if __name__ == '__main__':
    go()