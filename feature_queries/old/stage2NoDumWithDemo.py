import pandas as pd
import psycopg2
import sys

def go(output_fn):
    '''Generate dataframe with features from database'''

    conn = psycopg2.connect("dbname = police user = lauren password = llc")

    #Queries for features
    outcome = 'SELECT crid, officer_id, "Findings Sustained" FROM dependent_dum;'

    alleg = "SELECT crid, a.officer_id, beat, a.dateobj, i.investigator_id, o.race_edit AS officer_race, o.gender AS officer_gender, tractce10,\
                (CASE WHEN a.finding_edit = 'No Affidavit' THEN 1 ELSE 0 END) AS no_affidavit, \
                (CASE WHEN EXTRACT(dow FROM a.dateobj) NOT IN (0, 6) THEN 1 ELSE 0 END) AS weekend, \
                (CASE WHEN o.rank IS NOT NULL THEN o.rank ELSE 'UNKNOWN' END) AS rank, \
                (CASE WHEN investigator_name IN (SELECT concat_ws(', ', officer_last, officer_first) \
                FROM officers) THEN 1 ELSE 0 END) AS police_investigator, oc.centrality_score  \
                FROM allegations as a LEFT JOIN officers as o \
                ON (a.officer_id = o.officer_id) \
                LEFT JOIN investigators AS i ON (a.investigator_id = i.investigator_id) \
                LEFT JOIN officer_centralities AS oc ON (a.officer_id = oc.officer_id) \
                WHERE tractce10 IS NOT NULL;"

    age = "SELECT crid, officer_id, officers_age, (officers_age^2) AS agesqrd FROM ages;"

    data311 = "SELECT * FROM time_distance_311;"
    datacrime = "SELECT * FROM time_distance_crime;"
    other_complaints = "SELECT * FROM time_distance_complaints;"

    priors = "SELECT * FROM prior_complaints;"

    acs = "SELECT tract_1, pct017, pct1824, pct2534, pct3544, pct4554, pct5564, pct6500, \
                ptnla, ptnlb, ptnlwh, ptnloth, ptl, ptlths, pthsged, ptsomeco, ptbaplus, ptpov, pctfb \
                FROM acs;"

    complainant_demo = "SELECT crid, gender AS complainant_gender, race_edit AS complainant_race, \
                        age AS complainant_age from complainants;"

    witnesses = "SELECT crid, count(*) FROM witnesses GROUP BY crid;"

    phys = "SELECT * from physical_dummies;"

    travel_times = "SELECT crid, officer_id, t.car_time, t.transit_time FROM \
                    allegations AS a LEFT JOIN \
                        (SELECT tt.beat_num, tt.time / 60 AS transit_time, ct.time / 60 AS car_time, tt.end_address \
                        FROM car_times AS ct JOIN transit_times AS tt ON \
                        (ct.beat_num = tt.beat_num AND ct.end_address = tt.end_address)) AS t \
                    ON (a.beat::numeric = t.beat_num::numeric) \
                    WHERE ((a.incident_date < '01/01/2012'::date) AND t.end_address ~ '.*60616.*') \
                    OR ((a.incident_date >= '01/01/2012'::date) AND t.end_address ~ '.*60622.*');"

    outcome_df = pd.read_sql(outcome, conn)
    alleg_df = pd.read_sql(alleg, conn)
    age_df = pd.read_sql(age, conn)
    data311_df = pd.read_sql(data311, conn)
    datacrime_df = pd.read_sql(datacrime, conn)
    other_df = pd.read_sql(other_complaints, conn)
    priors_df = pd.read_sql(priors, conn)
    acs_df = pd.read_sql(acs, conn)
    complainants_df = pd.read_sql(complainant_demo, conn)
    witnesses_df = pd.read_sql(witnesses, conn)
    phys_df = pd.read_sql(phys, conn)
    travel_df = pd.read_sql(travel_times, conn)

    #Close connection to database after queries
    conn.commit()
    conn.close()

    #Drop duplicate geographic data
    data311_df.drop_duplicates('crid', inplace = True)
    datacrime_df.drop_duplicates('crid', inplace = True)
    acs_df.drop_duplicates(inplace = True)
    other_df.drop_duplicates('crid', inplace = True)

    #Merge (join) dataframes on shared keys
    df_final = outcome_df.merge(alleg_df, on = ['crid', 'officer_id'], how = 'right')\
                .merge(age_df, on = ['crid', 'officer_id'], how = 'left')\
                .merge(data311_df, on = 'crid', how = 'left').merge(datacrime_df, on = 'crid', how = 'left')\
                .merge(other_df, on = 'crid', how = 'left')\
                .merge(priors_df, on = ['crid', 'officer_id'], how = 'left')\
                .merge(acs_df, how = 'left', left_on = 'tractce10', right_on = 'tract_1')\
                .merge(complainants_df, how = 'left', on = 'crid')\
                .merge(witnesses_df, on = 'crid', how = 'left')\
                .merge(phys_df, on = ['crid', 'officer_id'], how = 'left')\
                .merge(travel_df, on = ['crid', 'officer_id'], how = 'left')

    df_final.drop(['tract_1'], axis = 1, inplace = True)

    df_final.to_csv(output_fn)

if __name__ == '__main__':

    go(sys.argv[1])