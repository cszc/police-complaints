import sys
from sqlalchemy import create_engine
import pandas as pd

def import_csv(dbname, f):
        chunks = 20000
        j=0
        index_start = 1
        engine = create_engine('postgresql://lauren:llc@localhost:5432/police')
        for df in pd.read_csv(f, chunksize = chunks, iterator=True, encoding='utf-8'):
                df.index += index_start
                j+=1
                print("completed {} rows".format(j*chunks))
                df.to_sql(dbname, engine, if_exists='append')
                index_start = df.index[-1] + 1

if __name__=="__main__":
        dbname = sys.argv[1]
        f = sys.argv[2]
        import_csv(dbname, f)