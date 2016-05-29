import googlemaps
import csv
import time
import json

with open('ambiguousUnresolved.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    origin_addresses = []
    for row in reader:
        address = row['address']
        crid = row['crid']
        origin_addresses.append((crid, address))

ambig_file = open('ambiguous_incident_addresses2.csv', 'a')
unambig_file = open('unambiguous_incident_addresses2.csv', 'a')
fieldnames = ['crid', 'address', 'geocode']
key_file = open('google_api_keys.txt', 'r')

ambig_writer = csv.DictWriter(ambig_file, fieldnames=fieldnames)
unambig_writer = csv.DictWriter(unambig_file, fieldnames=fieldnames)

for i, key in enumerate(key_file):
    print('starting new key')
    try:
        key = key.strip()
        gmaps = googlemaps.Client(key)
        for crid, address in origin_addresses:
            try:
                geocode = gmaps.geocode(address=address)
                if len(geocode) > 1:
                    ambig_writer.writerow(
                        {'crid': crid, 'address': address, 'geocode': geocode})
                else:
                    unambig_writer.writerow(
                        {'crid': crid, 'address': address, 'geocode': geocode})
            except Exception as err:
                print('Error: {}'.format(err))
                ambig_writer.writerow({'crid': crid, 'address': address})
                continue
    except Exception as e:
        print(e)
        continue

ambig_file.close()
unambig_file.close()
