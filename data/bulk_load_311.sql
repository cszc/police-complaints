ALTER TABLE sanitation ADD COLUMN geom geometry(POINT,4326);
UPDATE sanitation SET geom = ST_SetSRID(ST_MakePoint("Longitude","Latitude"),4326);
CREATE INDEX idx_sanitation_geom ON sanitation USING GIST(geom);
alter table sanitation add column dateobj TIMESTAMP;
update sanitation set dateobj = to_timestamp("Creation Date", 'MM/DD/YYYY');

ALTER TABLE alleylights ADD COLUMN geom geometry(POINT,4326);
UPDATE alleylights SET geom = ST_SetSRID(ST_MakePoint("Longitude","Latitude"),4326);
CREATE INDEX idx_alley_geom ON alleylights USING GIST(geom);
alter table alleylights add column dateobj TIMESTAMP;
update alleylights set dateobj = to_timestamp("Creation Date", 'MM/DD/YYYY');

ALTER TABLE vacantbuildings ADD COLUMN geom geometry(POINT,4326);
UPDATE vacantbuildings SET geom = ST_SetSRID(ST_MakePoint("LONGITUDE","LATITUDE"),4326);
CREATE INDEX idx_vacantbuildings_geom ON vacantbuildings USING GIST(geom);
alter table vacantbuildings add column dateobj TIMESTAMP;
update vacantbuildings set dateobj = to_timestamp("DATE SERVICE REQUEST WAS RECEIVED", 'MM/DD/YYYY');

ALTER TABLE streetlights_all ADD COLUMN geom geometry(POINT,4326);
UPDATE streetlights_all SET geom = ST_SetSRID(ST_MakePoint("Longitude","Latitude"),4326);
CREATE INDEX idx_streetall_geom ON streetlights_all USING GIST(geom);
alter table streetlights_all add column dateobj TIMESTAMP;
update streetlights_all set dateobj = to_timestamp("Creation Date", 'MM/DD/YYYY');

ALTER TABLE vehicles ADD COLUMN geom geometry(POINT,4326);
UPDATE vehicles SET geom = ST_SetSRID(ST_MakePoint("Longitude","Latitude"),4326);
CREATE INDEX idx_vehicles_geom ON vehicles USING GIST(geom);
alter table vehicles add column dateobj TIMESTAMP;
update vehicles set dateobj = to_timestamp("Creation Date", 'MM/DD/YYYY');

ALTER TABLE streetlights_one ADD COLUMN geom geometry(POINT,4326);
UPDATE streetlights_one SET geom = ST_SetSRID(ST_MakePoint("Longitude","Latitude"),4326);
CREATE INDEX idx_streetlights_one_geom ON streetlights_one USING GIST(geom);
alter table streetlights_one add column dateobj TIMESTAMP;
update streetlights_one set dateobj = to_timestamp("Creation Date", 'MM/DD/YYYY');


ALTER TABLE treetrims ADD COLUMN geom geometry(POINT,4326);
UPDATE treetrims SET geom = ST_SetSRID(ST_MakePoint("Longitude","Latitude"),4326);
CREATE INDEX idx_treetrims_geom ON treetrims USING GIST(geom);
alter table treetrims add column dateobj TIMESTAMP;
update treetrims set dateobj = to_timestamp("Creation Date", 'MM/DD/YYYY');

ALTER TABLE potholes ADD COLUMN geom geometry(POINT,4326);
UPDATE potholes SET geom = ST_SetSRID(ST_MakePoint("LONGITUDE","LATITUDE"),4326);
CREATE INDEX idx_potholes_geom ON potholes USING GIST(geom);
alter table potholes add column dateobj TIMESTAMP;
update potholes set dateobj = to_timestamp("CREATION DATE", 'MM/DD/YYYY');


ALTER TABLE graffiti ADD COLUMN geom geometry(POINT,4326);
UPDATE graffiti SET geom = ST_SetSRID(ST_MakePoint("Longitude","Latitude"),4326);
CREATE INDEX idx_graffiti_geom ON graffiti USING GIST(geom);
alter table graffiti add column dateobj TIMESTAMP;
update graffiti set dateobj = to_timestamp("Creation Date", 'MM/DD/YYYY');

drop table if exists radius311;
select crid, officer_id into radius311 from test2;