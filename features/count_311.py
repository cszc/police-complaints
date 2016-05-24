import psycopg2 as ps
from psycopg2.extensions import AsIs
from psycopg2.extensions import QuotedString


MAIN_TABLE = "311bytract"
# TABLES_TO_COUNT = ['311alleylights',"311garbage", "311graffiti", "311potholes", "311rodent", "311sanitation", "311streetlightsall", "311streetlightsone", "311trees", "311vap","311vehicles"]

TABLES_TO_COUNT = ['311alleylights']

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

    def count_311_calls(self, table311):
        name = str(table311)
        print("Starting {}".format(table311))
        cur = self.dbconn.cursor()
        # add a geometry column to an existing 311 table table
        sql = "ALTER TABLE \"" + table311 + "\" ADD COLUMN geom geometry(POINT,4326);"
        cur.execute(sql)
        
        # make a geopoint column from existing text lat & long columns
        # note that for some reason lat/lng are reverse from what you'd expect
        sql = "UPDATE \"" + table311 + "\" SET geom = ST_SetSRID(ST_MakePoint(lng,lat),4326);"
        cur.execute(sql)

        index_name = "idx_" + name + "_geom"
        # make an index
        "CREATE INDEX "+ index_name + " ON \""+table311+"\" USING GIST(geom);"
        cur.execute(sql)
         
        # count how many 311 calls there were per census tract and add that to the main table

        col_name = name + "_count"
        sql = "alter \""+ MAIN_TABLE + "\" add "+col_name+" int;"
        cur.execute(sql)

        sql = "update \""+ MAIN_TABLE+"\" set "+ col_name +" = b.cnt from (select count(*) as cnt, t.tractce10 as tract from \""+table311+"\"as complaints JOIN "tracts2010" as t on ST_Contains(t.geom, complaints.geom) group by t.tractce10) as b where tractce10 = b.tract;"
        cur.execute(sql)
        self.dbconn.commit()
        print("Completed {}".format(name))
        cur.close()

if __name__ == "__main__":
    dbClient = client()
    try:            
        dbClient.openConnection()
    except Exception as e:
        print("Error: {}".format(e))

    for table in TABLES_TO_COUNT:
        dbClient.count_311_calls(table)

    dbClient.closeConnection()

