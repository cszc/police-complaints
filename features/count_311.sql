import psycopg2 as ps

class client:
    def __init__(self):
        # you add class variables here like
        # self.myvar="the greatest variable ever. the best"
        self.dbname="police"
        self.dbhost="localhost"
        self.dbport=5432
        self.dbusername="lauren"
        self.dbpasswd="llc"
        self.dbconn=None
    # open a connection to a psql database
    def openConnection(self):
        #
        print "Opening a Connection"
        conn = ps.connect(database=self.dbname, user=self.dbusername, password=self.dbpasswd,\
            host=self.dbhost, port=self.dbport)
        self.dbconn = conn

    # Close any active connection(should be able to handle closing a closed conn)
    def closeConnection(self):
        print "Closing Connection"

        if self.dbconn:
            self.dbconn.close()
            self.dbconn=None
            print "Connection closed"
        else:
            print "Connection already closed"

MAIN_TABLE = "311bytract"

# add a geometry column to an existing 311 table table
ALTER TABLE "311rodent" ADD COLUMN geom geometry(POINT,4326);
# make a geopoint column from existing text lat & long columns
# note that for some reason lat/lng are reverse from what you'd expect
UPDATE "311rodent" SET geom = ST_SetSRID(ST_MakePoint(lng,lat),4326);
# make an index
CREATE INDEX idx_311rodent_geom ON "311rodent" USING GIST(geom);
 
# count how many 311 rodent calls there were per census tract
select count(*), t.tractce10 from "311rodent" as r join "tracts2010" as t on ST_Contains(t.geom, r.geom) group by t.tractce10;

alter table A add column3 datatype

update A 
set column3 = B.column3 
from A inner join B on A.Column1 = B.Column2