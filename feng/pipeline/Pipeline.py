from sklearn.pipeline import Pipeline as _Pipeline
import pandas as pd


class Pipeline(_Pipeline):

    def __init__(self, steps, fields=None, **kwargs):
        super(Pipeline, self).__init__(steps, **kwargs)
        self.fields = fields

    def fit(self, X, y=None, **fit_params):
        super(Pipeline, self).fit(_select(X, self.fields), y, **fit_params)
        return self

    def transform(self, X):
        return super(Pipeline, self).transform(_select(X, self.fields))

    def fit_transform(self, X, y=None, **fit_params):
        return super(Pipeline, self).fit_transform(_select(X, self.fields), y, **fit_params)


def _select(X, fields):
    if isinstance(X, pd.DataFrame):
        return X[fields] if fields is not None else X
    else:
        return X[:, fields] if fields is not None else X
