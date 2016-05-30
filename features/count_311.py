import psycopg2 as ps
from psycopg2.extensions import AsIs


# TABLES_311 = ["311alleylights","311garbage", "311graffiti", "311potholes", "311rodent", "311sanitation", "311streetlightsall", "311streetlightsone", "311trees", "311vap","311vehicles"]

# NEW_311 = ["rodents","garbage","sanitation", "alleylights", "vacantbuildings", "streetlights_all", "vehicles", "streetlights_one", "treetrims", "potholes", "graffiti"]

NEW_311 = ["treetrims", "potholes", "graffiti"]

CRIMES = ["crimetest"]

FBI_CODES = ["18", "08A", "02", "08B", "17", "16", "03", "01B", "24", "06", "07", "19", "04A", "11", "10", "12", "05", "09", "13", "26", "01A", "14", "04B", "15", "20", "22"]

PARTICIPANT_TABLES = ["officers"]
ALLEGATIONS_TABLE = "allegations"
DISTANCES = ['500','1000', '2500']
TIMES = ['7 days', '3 months']

class client:
    def __init__(self):
       
        self.dbname="police"
        self.dbhost="localhost"
        self.dbport=5432
        self.dbusername="lauren"
        self.dbpasswd="llc"
        self.dbconn=None
    # open a connection to a psql database
    def openConnection(self):
        #
        print("Opening a Connection")
        conn = ps.connect(database=self.dbname, user=self.dbusername, password=self.dbpasswd,\
            host=self.dbhost, port=self.dbport)
        self.dbconn = conn

    # Close any active connection(should be able to handle closing a closed conn)
    def closeConnection(self):
        print("Closing Connection")

        if self.dbconn:
            self.dbconn.close()
            self.dbconn=None
            print("Connection closed")
        else:
            print("Connection already closed")

    def add_coords(self, tablename):
        name = "\""+str(tablename)+"\""
        # main_name = "\""+str(MAIN_TABLE)+"\""
        print("Starting {}".format(tablename))
        cur = self.dbconn.cursor()
        # add a geometry column to an existing 311 table table
        cur.execute('''
            ALTER TABLE %s ADD COLUMN geom geometry(POINT,4326);
            ''', [AsIs(name)])
        print("success #1")
        # make a geopoint column from existing text lat & long columns
        # note that for some reason lat/lng are reverse from what you'd expect
        cur.execute("UPDATE %s SET geom = ST_SetSRID(ST_MakePoint(lng,lat),4326);", [AsIs(name)])
        print("success #2")
        index_name = "idx_" + tablename + "_geom"
        # make an index
        cur.execute("CREATE INDEX %s ON %s USING GIST(geom);", (AsIs(index_name), AsIs(name)))
        print("success #3")

        self.dbconn.commit()
        cur.close()


    def make_new_feature_table(self, allegations, out_table):
        cur = self.dbconn.cursor()
        cur.execute("select crid into %s from %s;",(AsIs(out_table),AsIs(allegations)))
        dbClient.add_index_crid(out_table)
        self.dbconn.commit()
        print("Created table".format(out_table))
        cur.close()

    def get_crimes_by_radii(self, allegations, crimetable, out_table):
        print("Starting {}".format(crimetable))
        # out_table = "radius311"
        # out_table = "radiuscrime"
        
        # distances = ['500','1000', '2500', '5000']
        # times = ['7 days', '30 days','3 months','6 months', '1 year']
        
        for d in DISTANCES:
            for time in TIMES:
                for code in FBI_CODES:
                    cur = self.dbconn.cursor()

                    print("starting {} crimes, {}, {} m".format(code, time, d))
                    
                    col_name = "crimes_" + code + "_"+ time.replace(" ","") + d + 'm'
                    cur.execute("alter table %s add column %s int;", (AsIs(out_table), AsIs(col_name)))
                    print("added col {}".format(col_name))
                    
                    cur.execute(
                        '''
                        update %s
                        set %s = agg.num_complaints
                        from
                        (SELECT a.crid, COUNT(*) as num_complaints
                        FROM %s as a JOIN %s as b
                        ON ST_DWithin(a.geom::geography, b.geom::geography, %s)
                        AND b.dateobj < a.dateobj
                        AND b.dateobj > (a.dateobj - interval '%s')
                        WHERE b."FBI Code" = %s
                        GROUP BY a.crid) as agg
                        where %s.crid=agg.crid;
                        ''', (AsIs(out_table),AsIs(col_name),AsIs(allegations),AsIs(crimetable),AsIs(d),AsIs(time),str(code),AsIs(out_table)))
                    print("Completed query")
                    self.dbconn.commit()
                    cur.close()
        print("Completed counting {}".format(crimetable))



    def get_311_radii(self, allegations, table311, out_table):
        print("Starting {}".format(table311))
        # out_table = "radius311"
        # out_table = "radiuscrime"
        # cur = self.dbconn.cursor()
        dist2 = ['500','1000', '2500']
        times2 = ['7 days', '3 months']
        
        for d in dist2:
            for time in times2:
                cur = self.dbconn.cursor()


                print("starting {}, {} m".format(time, d))
                
                col_name = table311 + "_count_" + time.replace(" ","") + d + 'm'
                cur.execute("alter table %s add column %s int;", (AsIs(out_table), AsIs(col_name)))
                print("added col {}".format(col_name))
                
                cur.execute(
                    '''
                    update %s
                    set %s = agg.num_complaints
                    from
                    (SELECT a.crid, COUNT(*) as num_complaints
                    FROM %s as a JOIN %s as b
                    ON ST_DWithin(a.geom::geography, b.geom::geography, %s)
                    AND b.dateobj < a.dateobj
                    AND b.dateobj > (a.dateobj - interval '%s')
                    GROUP BY a.crid) as agg
                    where %s.crid=agg.crid;
                    ''', (AsIs(out_table),AsIs(col_name),AsIs(allegations),AsIs(table311),AsIs(d),AsIs(time),AsIs(out_table)))
                print("Completed query")
                self.dbconn.commit()
                cur.close()

        print("Completed counting {}".format(table311))
# # '''
# alter table allegations add column tractce10 int;
# update allegations
# set tractce10 = agg.tractce10
# from
# (select distinct a.crid, t.tractce10
# from allegations as a join tracts2010 as t
# on ST_Contains(t.geom, a.geom)) as agg
# where allegations.crid=agg.crid;
# # '''
    def count_other_complaints(self, allegations, out_table):
        print("Starting {}".format(allegations))
            # out_table = "radius311"
            # out_table = "radiuscrime"
        
        
        for d in DISTANCES:
            for time in TIMES:
                cur = self.dbconn.cursor()

                print("starting {}, {} m".format(time, d))
                
                col_name = "allegationcount" + time.replace(" ","") + d + 'm'
                cur.execute("alter table %s add column %s int;", (AsIs(out_table), AsIs(col_name)))
                print("added col {}".format(col_name))
                
                cur.execute(
                    '''
                    update %s
                    set %s = agg.num_complaints
                    from
                    (SELECT a.crid, COUNT(*) as num_complaints
                    FROM %s as a JOIN %s as b
                    ON ST_DWithin(a.geom::geography, b.geom::geography, %s)
                    AND b.dateobj < a.dateobj
                    AND b.dateobj > (a.dateobj - interval '%s')
                    GROUP BY a.crid) as agg
                    where %s.crid=agg.crid;
                    ''', (AsIs(out_table),AsIs(col_name),AsIs(allegations),AsIs(allegations),AsIs(d),AsIs(time),AsIs(out_table)))
                print("Completed query")
                self.dbconn.commit()
                cur.close()
        print("Completed counting {}".format(allegations))




    def count_311_calls(self, table311):
        name = "\""+str(table311)+"\""
        # main_name = "\""+str(MAIN_TABLE)+"\""
        print("Starting {}".format(table311))
        cur = self.dbconn.cursor()


        # count how many 311 calls there were per census tract and add that to the main table

        col_name = "\""+table311 + "_count\""
        cur.execute("alter table \"311bytract\" add column %s int;", [AsIs(col_name)])
        print("success #4")

        cur.execute("""
        update \"311bytract\"
        set %s = b.cnt
        from ( 
            select count(*) as cnt, t.tractce10 as tract from %s as complaints 
            JOIN \"tracts2010\" as t 
            on ST_Contains(t.geom, complaints.geom) 
            group by t.tractce10) as b
        where tractce10 = b.tract;
        """, (AsIs(col_name), AsIs(name)))
        self.dbconn.commit()
        print("Completed {}".format(name))
        cur.close()


    def get_participant_age(self, allegations, participant_table, out_table):
        col_name = participant_table + "_age"
        cur = self.dbconn.cursor()

        cur.execute("alter table %s add column %s int;", (AsIs(out_table), AsIs(col_name)))
        print("added col {} to {}".format(col_name, out_table))

        cur.execute('''
        update %s
        set %s = agg.age 
        from (SELECT (a.crid, a.officer_id) AS allegation_id, extract(year from age(a.dateobj, b.dateobj)) as age
            FROM %s as a JOIN %s as b
            ON a.officer_id=b.officer_id) as agg
        where (crid, officer_id)=agg.allegation_id;
        ''',(AsIs(out_table), AsIs(col_name), AsIs(allegations), AsIs(participant_table)))
        self.dbconn.commit()
        print("Completed {} age".format(participant_table))
        cur.close()

    def add_index_crid(self, table):
        cur = self.dbconn.cursor()
        ix_name = "idx_" + table + "_crid"
        cur.execute("CREATE INDEX %s ON %s(crid);", (AsIs(ix_name), AsIs(table)))
        self.dbconn.commit()
        cur.close()



if __name__ == "__main__":
    dbClient = client()
    try:            
        dbClient.openConnection()
    except Exception as e:
        print("Error: {}".format(e))

    # for table in TABLES_311:
    #     dbClient.count_311_calls(table)

    # for table in NEW_311:
    #     dbClient.get_311_radii("test2",table,"radius311")
  

    #Count 311
    # results311 = "time_distance_311"
    # # dbClient.make_new_feature_table(ALLEGATIONS_TABLE, results311)
    # print("Created {}".format(results311))
    # for table in NEW_311:
    #     print("Starting {}".format(table))

    #     dbClient.get_311_radii(ALLEGATIONS_TABLE, table, results311)

    # #Count crimes
    # resultscrime = "time_distance_crime"
    # dbClient.make_new_feature_table(ALLEGATIONS_TABLE, resultscrime)
    # print("Created {}".format(resultscrime))
    # for table in CRIMES:
    #     print("Starting {}".format(table))

    #     dbClient.get_crimes_by_radii(ALLEGATIONS_TABLE, table, resultscrime)

    #count other complaints
    resultscomplaints = "time_distance_complaints"
    # # dbClient.make_new_feature_table(ALLEGATIONS_TABLE, resultscomplaints)
    # print("Created {}".format(resultscomplaints))
    # print("Starting aggregate {}".format(ALLEGATIONS_TABLE))

    dbClient.count_other_complaints(ALLEGATIONS_TABLE, resultscomplaints)

    #calculate ages
    resultsage = "ages"
    print("Starting ages")
    for p in PARTICIPANT_TABLES:
        dbClient.get_participant_age(ALLEGATIONS,p, out_table)

    dbClient.closeConnection()

