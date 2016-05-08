import psycopg2
import os
import csv
from argparse import ArgumentParser

def main():
    '''
    Takes a directory as an argument.
    Copies csvs in that directory to posgres db.
    '''
    parser = ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    onet_to_psql(args.directory)


SQL_STATEMENT = """
    COPY %s FROM STDIN WITH
        CSV
        HEADER
        DELIMITER AS ','
    """


def process_file(conn, table_name, file_object):
    cur = conn.cursor()
    cur.copy_expert(sql=SQL_STATEMENT % table_name, file=file_object)
    conn.commit()
    cursor.close()


def onet_to_psql(directory):
    conn = psycopg2.connect(
        "host='localhost' port='5432' dbname='police' user='christine' password='llc'")
    for entry in os.scandir(path=directory):
        f = open(os.path.join(directory, entry.name))
        try:
            process_file(conn, entry.name, f)
        except:
            print("something did'nt work")

if __name__ == '__main__':
    main()
