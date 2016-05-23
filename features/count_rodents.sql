ALTER TABLE "311rodent" ADD COLUMN geom geometry(POINT,4326);
UPDATE "311rodent" SET geom = ST_SetSRID(ST_MakePoint(lng,lat),4326);
CREATE INDEX "idx_rodent_geom" ON "311rodent" USING GIST(geom);
ALTER TABLE "311bytract" ADD rodent_count int;
update "311bytract" 
set rodent_count = ( 
    select count(*) as cnt from "311rodent" as complaints 
    JOIN "tracts2010" as t 
    on ST_Contains(t.geom, complaints.geom) 
    group by t.tractce10);