from sklearn.pipeline import Pipeline


class SelectivePipeline(Pipeline):

    def __init__(self, steps, fields=None):
        super(SelectivePipeline, self).__init__(steps)
        self.fields = fields

    def fit(self, X, y=None, **fit_params):
        if not self.fields:
            fields = X.columns.values
        super(SelectivePipeline, self).fit(X[self.fields], y, **fit_params)
        return self

    def transform(self, X):
        return super(SelectivePipeline, self).transform(X[self.fields])

    def fit_transform(self, X, y=None, **fit_params):
        return super(SelectivePipeline, self).fit_transform(X[self.fields], y, **fit_params)
