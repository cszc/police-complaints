import pandas as pd
import ast
import sys

def go(fn):
    df = pd.read_csv(fn, index_col = 0)

    def latfunc(x):
        dic = ast.literal_eval(x)[0]
        lat = dic['geometry']['location']['lat']
        return lat

    def lngfunc(x):
        dic = ast.literal_eval(x)[0]
        lng = dic['geometry']['location']['lng']
        return lng

    df_final = pd.DataFrame(index = df.index)
    df_final["lat"] = df.geocode.apply(latfunc)
    df_final["lng"] = df.geocode.apply(lngfunc)

    df_final.to_csv("unambiguousResolved2.csv")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Missing file name")
    else:
        go(sys.argv[1])
