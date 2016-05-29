import googlemaps
import csv
import time
import json

with open('allegations_addresses.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    origin_addresses = []
    for row in reader:
        address = '{} {}, {}'.format(
            row['add1'], row['add2'], row['city'])
        crid = row['crid']
        origin_addresses.append((crid, address))

start = 9213
BATCH = 2245

ambig_file = open('ambiguous_incident_addresses.csv', 'a')
unambig_file = open('unambiguous_incident_addresses.csv', 'a')
fieldnames = ['crid', 'address', 'geocode']
key_file = open('google_api_keys.txt', 'r')

ambig_writer = csv.DictWriter(ambig_file, fieldnames=fieldnames)
unambig_writer = csv.DictWriter(unambig_file, fieldnames=fieldnames)

for i, key in enumerate(key_file):
    print('starting new key')
    try:
        key = key.strip()
        gmaps = googlemaps.Client(key)
        for crid, address in origin_addresses[start:start + BATCH]:
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
        start += BATCH
    except Exception as e:
        print(e)
        continue

print('Next time, start at: {0}'.format(start))
ambig_file.close()
unambig_file.close()
