import pandas as pd

def go():
    df = pd.read_csv("../data/allegations_clean.csv")
    cat_df = pd.read_csv("../data/categories.csv")
    ids = cat_df.cat_id.tolist()
    physical = cat_df.physical.tolist()

    df.set_index(['crid', 'officer_id'], inplace = True)

    df.replace(to_replace=ids, value=physical, inplace=True)

    cat_dum = pd.get_dummies(df.cat_id, prefix = "physical", prefix_sep = ":", dummy_na = True)

    cat_dum.to_csv("catPhysDummyFeatures.csv")

if __name__ == '__main__':
    go()
