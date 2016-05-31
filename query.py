import pandas as pd
import psycopg2

def go():
    '''Generate dataframe with features from database'''

    conn = psycopg2.connect("dbname = police user = lauren password = llc")

    #Queries for features
    alleg = "SELECT crid, officer_id, tractce10, pct017, pct1824, pct2534, pct3544, pct4554, pct5564, pct6500, \
                ptnla, ptnlb, ptnlwh, ptnloth, ptl, ptlths, pthsged, ptsomeco, ptbaplus, ptpov, pctfb \
                FROM allegations JOIN acs ON (allegations.tractce10 = acs.tract_1);"

    invest1 = "SELECT * FROM investigator_beat_dum1;"
    invest2 = "SELECT * FROM investigator_beat_dum2;"

    age = "SELECT crid, officer_id, officers_age FROM ages;"

    data311 = "SELECT * FROM time_distance_311;"

    acs = "SELECT tract_1, pct017, pct1824, pct2534, pct3544, pct4554, pct5564, pct6500, \
                ptnla, ptnlb, ptnlwh, ptnloth, ptl, ptlths, pthsged, ptsomeco, ptbaplus, ptpov, pctfb \
                FROM acs;"

    alleg_df = pd.read_sql(alleg, conn)
    invest1_df = pd.read_sql(invest1, conn)
    invest2_df = pd.read_sql(invest2, conn)
    age_df = pd.read_sql(age, conn)
    data311_df = pd.read_sql(data311, conn)
    #Close connection to database after making queries
    conn.commit()
    conn.close()
    #Merge (join) dataframes on shared keys
    df_final = alleg_df.merge(invest1_df, on = ['crid', 'officer_id'], how = 'left')\
                .merge(invest2_df, on = ['crid', 'officer_id'], how = 'left')\
                .merge(age_df, on = ['crid', 'officer_id'], how = 'left')

    df_final = df_final.join(data311_df.drop('crid', axis = 1))
    #Drop sequential index column
    df_final.drop('index', axis = 1, inplace = True)

    return df_final

if __name__ == '__main__':
    go()
