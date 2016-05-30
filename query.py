import pandas as pd
import psycopg2

def go():
    conn = psycopg2.connect("dbname = police user = lauren password = llc")

    alleg = "SELECT crid, officer_id, tractce10 FROM allegations;"

    invest = "SELECT * FROM investigator_beat_dum1 NATURAL JOIN investigator_beat_dum2;"

    age = "SELECT crid, officer_id, officers_age FROM ages;"

    data311 = "SELECT * FROM time_distance_311;"

    acs = "SELECT tract_1, pct017, pct1824, pct2534, pct3544, pct4554, pct5564, pct6500, \
                ptnla, ptnlb, ptnlwh, ptnloth, ptl, ptlths, pthsged, ptsomeco, ptbaplus, ptpov, pctfb \
                FROM acs;"

    alleg_df = pd.read_sql(alleg, conn)
    invest_df = pd.read_sql(invest, conn)
    age_df = pd.read_sql(age, conn)
    data311_df = pd.read_sql(data311, conn)
    acs_df = pd.read_sql(acs, conn)

    conn.commit()
    conn.close()

    # data311_df.rename(columns = {'tract_1' : 'tractce10'}, inplace = True)

    df_final1 = alleg_df.join(invest_df, on =['crid', 'officer_id'], how = 'left')#.join(age_df, on = ['crid', 'officer_id'], how = 'left')
    #df_final = df_final1.join(data311, on = 'crid', how = 'left').join(acs_df, how = 'left', left_on = 'tractce10', right_on = 'tract_1')


    return df_final1
