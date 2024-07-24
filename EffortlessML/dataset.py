from sklearn import datasets

from EffortlessML.data import MLData


def get_breast_cancer():
    '''
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    '''
    dataset = datasets.load_breast_cancer(as_frame=True)
    x_cols = list(dataset['feature_names'])
    y_col = 'target'
    return MLData(df=dataset['frame'], x_cols=x_cols, y_col=y_col)
