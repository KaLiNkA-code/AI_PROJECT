from sklearn import preprocessing


def preprocess(DataSet):
    feature_names = DataSet.columns.tolist() 
    for column in feature_names: 
        print(column)
    DataSet[column].value_counts(dropna=False) 
    DataSet['Annual Income (k$)'] = preprocessing.normalize(DataSet['Annual Income (k$)'].reshape(-1, 1))
    DataSet['Spending Score (1-100)'] = preprocessing.normalize(DataSet['Spending Score (1-100)'].reshape(-1, 1))
    return DataSet
