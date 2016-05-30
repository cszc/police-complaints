import pandas as pd

def go():
    df = pd.read_csv("../data/allegations_clean.csv")

    df.officer_id.fillna(0, inplace = True)
    df.set_index(['crid', 'officer_id'], inplace = True)

    def bucket(x):
        if x.isnumeric():
           if int(x) >= 30:
               return "Over 30"
           else:
               return "Less than 30"
        elif x == "Over 30":
            return "Over 30"
        else:
            return x

    df.outcome_edit = df.outcome_edit.apply(bucket)

    findings_dum = pd.get_dummies(df.finding_edit, prefix = "Findings", prefix_sep = " ", dummy_na = True)
    outcome_dum = pd.get_dummies(df.outcome_edit, prefix = "Outcomes", prefix_sep = " ", dummy_na = True)

    investigators_dum = pd.get_dummies(df.investigator_id, prefix = "Investigators", prefix_sep = " ", dummy_na = True)
    beats = pd.get_dummies(df.beat, prefix = "Beat", prefix_sep = " ", dummy_na = True)

    dep_dummies = findings_dum.join(outcome_dum)

    ind_dummies = investigators_dum.join(beats)
    if len(ind_dummies.columns) > 1500:
        ind_dummies[ind_dummies.columns[:800]].to_csv("indDummyVar1.csv")
        ind_dummies[ind_dummies.columns[:800]].to_csv("indDummyVar2.csv")
    else:
        ind_dummies.to_csv("indDummyVar.csv")

    dep_dummies.to_csv("depDummyVar.csv")


if __name__ == '__main__':
    go()
