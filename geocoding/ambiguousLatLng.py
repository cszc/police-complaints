import pandas as pd
import ast

def go():
    def latfunc(x):
        dic = ast.literal_eval(x)[0]
        lat = dic['geometry']['location']['lat']
        return lat

    def lngfunc(x):
        dic = ast.literal_eval(x)[0]
        lng = dic['geometry']['location']['lng']
        return lng

    df = pd.read_csv("ambiguous_incident_addresses.csv", index_col = 0)

    ambiguous = df[df.geocode.isnull()].drop("geocode", axis = 1)
    ambiguous.to_csv("ambiguousUnresolved.csv")

    partial = df[df.geocode.notnull()]

    df_final = pd.DataFrame(index = partial.index)
    df_final["lat"] = partial.geocode.apply(latfunc)
    df_final["lng"] = partial.geocode.apply(lngfunc)

    df_final.to_csv("ambiguousResolved.csv")

if __name__ == '__main__':
    go()
