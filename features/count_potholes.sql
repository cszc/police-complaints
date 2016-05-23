ALTER TABLE "311potholes" ADD COLUMN geom geometry(POINT,4326);
UPDATE "311potholes" SET geom = ST_SetSRID(ST_MakePoint(lng,lat),4326);
CREATE INDEX "idx_geom" ON "311potholes" USING GIST(geom);
ALTER TABLE "311bytract" ADD pothole_count int;
update "311bytract"
set rodent_count = b.cnt
from ( 
    select count(*) as cnt, t.tractce10 as tract from "311potholes" as complaints 
    JOIN "tracts2010" as t 
    on ST_Contains(t.geom, complaints.geom) 
    group by t.tractce10) as b
where tractce10 = b.tract;