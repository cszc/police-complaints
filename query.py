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


    for query in [alleg, invest, age, data311, acs]:
        str(query) + "_df" = pd.read_sql(query, conn)

    data311_df.rename(columns = {'tract_1' : 'tractce10'}, inplace = True)

    conn.commit()
    conn.close()

    df_final1 = alleg_df.join(invest_df, on =['crid', 'officer_id']).join(age_df, on = ['crid', 'officer_id'])
    df_final = df_final1.join(data311, on = 'crid').join(acs_df, on 'tractce10')


    return df_final
