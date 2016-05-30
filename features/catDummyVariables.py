import pandas as pd

def go():
    df = pd.read_csv("../data/allegations_clean.csv")
    cat_df = pd.read_csv("../data/categories.csv")
    ids = cat_df.cat_id.tolist()
    categories = cat_df.category.tolist()

    df.set_index(['crid', 'officer_id'], inplace = True)

    df.replace(to_replace=ids, value=categories, inplace=True)

    cat_dum = pd.get_dummies(df.cat_id, prefix = "Category", prefix_sep = " ", dummy_na = True)

    cat_dum.to_csv("catDummyVariableFeatures.csv")

if __name__ == '__main__':
    go()
