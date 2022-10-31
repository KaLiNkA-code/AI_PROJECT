from sklearn import preprocessing


def preprocess(DataSet):
    feature_names = DataSet.columns.tolist()
    for column in feature_names:
        print(column)
    DataSet[column].value_counts(dropna=False)
    DataSet["normalized_income"] = preprocessing.normalize(DataSet["Annual Income (k$)"]).flatten()
    DataSet["normalized_score"] = preprocessing.normalize(DataSet["Spending Score (1-100)"]).flatten()
    return DataSet
