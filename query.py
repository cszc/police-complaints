import pandas as pd
import pyscopg2 as pg2

def go():
    conn = psycopg2.connect("dbname = police user = lauren password = llc")


    invest_sel = "(SELECT * FROM investigator_beat_dum1 JOIN investigator_beat_dum2) as dummies \
                ON ((alleg.crid, alleg.officer_id) = (dummies.crid, dummies.officer_id))"

    age_sel = "(SELECT crid, officer_id, officers_age from ages) as age \
                ON ((alleg.crid, alleg.officer_id) = (age.crid, age.officer_id))"

    data311_sel = "(SELECT * FROM time_distance_311) as d311 \
                ON (alleg.crid = d311.crid)"

    acs_sel = "(SELECT tract_1, pct017, pct1824, pct2534, pct3544, pct4554, pct5564, pct6500, \
                ptnla, ptnlb, ptnlwh, ptnloth, ptl, ptlths, pthsged, ptsomeco, ptbaplus, ptpov, pctfb \
                FROM acs) as acs \
                ON (alleg.tractce10 = acs.tract_1)"

    query = "SELECT * FROM (SELECT crid, officer_id, tractce10, FROM allegations_clean) as alleg JOIN \
            {} JOIN {} JOIN {} JOIN {};".format(invest_sel, age_sel, data311_sel, acs_sel)"

    df = pd.read_sql(query, conn, index_col = ['crid', 'officer_id'])

    return df
