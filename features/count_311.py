import psycopg2 as ps
from psycopg2.extensions import AsIs


# TABLES_311 = ["311alleylights","311garbage", "311graffiti", "311potholes", "311rodent", "311sanitation", "311streetlightsall", "311streetlightsone", "311trees", "311vap","311vehicles"]

NEW_311 = ["rodents","garbage","sanitation", "alleylights", "vacantbuildings", "streetlights_all", "vehicles", "streetlights_one", "treetrims", "potholes", "graffiti"]

CRIMES = ["crimetest"]

FBI_CODES = ["18", "08A", "02", "08B", "17", "16", "03", "01B", "24", "06", "07", "19", "04A", "11", "10", "12", "05", "09", "13", "26", "01A", "14", "04B", "15", "20", "22"]

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

    def add_coords(self, table311):
        name = "\""+str(table311)+"\""
        # main_name = "\""+str(MAIN_TABLE)+"\""
        print("Starting {}".format(table311))
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
        index_name = "idx_" + table311 + "_geom"
        # make an index
        cur.execute("CREATE INDEX %s ON %s USING GIST(geom);", (AsIs(index_name), AsIs(name)))
        print("success #3")

        self.dbconn.commit()

    def make_new_feature_table(self, allegations, out_table):
        cur = self.dbconn.cursor()
        cur.execute("select crid, officer_id into %s from %s;",(AsIs(out_table),AsIs(allegations)))
        self.dbconn.commit()
        print("Created table".format(out_table))
        cur.close()

    def get_crimes_by_radii(self, allegations, crimetable, out_table):
        print("Starting {}".format(crimetable))
        # out_table = "radius311"
        # out_table = "radiuscrime"
        cur = self.dbconn.cursor()
        
        distances = ['1000', '2500', '5000']
        times = ['7 days', '30 days','6 months', '1 year']
        
        for d in distances:
            for time in times:
                for code in FBI_CODES:
                    print("starting {} crimes, {}, {} m".format(code, time, d))
                    
                    col_name = "crimes_" + code + "_"+ time.replace(" ","") + d + 'm'
                    cur.execute("alter table %s add column %s int;", (AsIs(out_table), AsIs(col_name)))
                    print("added col {}".format(col_name))
                    
                    cur.execute(
                        '''
                        update %s
                        set %s = agg.num_complaints
                        from
                        (SELECT (a.crid, a.officer_id) as allegation_id, COUNT(*) as num_complaints
                        FROM %s as a JOIN %s as b
                        ON ST_DWithin(a.geom::geography, b.geom::geography, %s)
                        AND b.dateobj < a.dateobj
                        AND b.dateobj > (a.dateobj - interval '%s')
                        WHERE b."FBI Code" == %s
                        GROUP BY (a.crid, a.officer_id)) as agg
                        where (crid, officer_id)=agg.allegation_id;
                        ''', (AsIs(out_table),AsIs(col_name),AsIs(allegations),AsIs(crimetable),AsIs(d),AsIs(time),AsIs(code)))
                    print("Completed query")
        self.dbconn.commit()
        print("Completed counting {}".format(crimetable))
        cur.close()


    def get_311_radii(self, allegations, table311, out_table):
        print("Starting {}".format(table311))
        # out_table = "radius311"
        # out_table = "radiuscrime"
        cur = self.dbconn.cursor()
        
        distances = ['500','1000','2000','2500', '5000']
        times = ['7 days', '14 days', '30 days', '3 months','6 months', '1 year']
        
        for d in distances:
            for time in times:
                print("starting {}, {} m".format(time, d))
                
                col_name = table311 + "_count_" + time.replace(" ","") + d + 'm'
                cur.execute("alter table %s add column %s int;", (AsIs(out_table), AsIs(col_name)))
                print("added col {}".format(col_name))
                
                cur.execute(
                    '''
                    update %s
                    set %s = agg.num_complaints
                    from
                    (SELECT (a.crid, a.officer_id) as allegation_id, COUNT(*) as num_complaints
                    FROM %s as a JOIN %s as b
                    ON ST_DWithin(a.geom::geography, b.geom::geography, %s)
                    AND b.dateobj < a.dateobj
                    AND b.dateobj > (a.dateobj - interval '%s')
                    GROUP BY (a.crid, a.officer_id)) as agg
                    where (crid, officer_id)=agg.allegation_id;
                    ''', (AsIs(out_table),AsIs(col_name),AsIs(allegations),AsIs(table311),AsIs(d),AsIs(time)))
                print("Completed query")
        self.dbconn.commit()
        print("Completed counting {}".format(table311))
        cur.close()




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
    allegations_table = "test2"
    out_table = "radiuscrime"
    # dbClient.make_new_feature_table(allegations_table, out_table)
    for crime_table in CRIMES:
        # dbClient.get_311_radii(allegations_table, crime_table, out_table)
        dbClient.get_crimes_by_radii(allegations_table,crime_table, out_table)

    dbClient.closeConnection()

