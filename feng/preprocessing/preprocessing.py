import pandas as pd
from sklearn.pipeline import TransformerMixin

class DateTimeFeatures(TransformerMixin):
    """Decomposes a set of date columns into numerical features.
    Properties you can consider adding to parameter 'props':
        'dayofyear',
        'daysinmonth',
        'is_month_end', 'is_month_start',
        'is_quarter_end', 'is_quarter_start',
        'is_year_end', 'is_year_start',
        'microsecond', 'nanosecond', 'second',
        'week'
    """
    def __init__(self, date_cols=[], props = ["year", "month", "day", "hour", "minute", "dayofweek", "quarter"]):
        self.date_cols = date_cols
        self.props = props

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.date_cols:
            X[c] = pd.to_datetime(X[c])
            for prop in self.props:
                X[c + "_" + prop] = getattr(X[c].dt, prop)
        X.drop(self.date_cols, axis=1, inplace=True)
        return X