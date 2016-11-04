import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import FeatureUnion as _FeatureUnion, _fit_transform_one, _transform_one
from sklearn.externals.joblib import Parallel, delayed


class FeatureUnion(_FeatureUnion):
    
    def fit_transform(self, X, y=None, **fit_params):
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, name, X, y, self.transformer_weights, **fit_params)
            for name, trans in self.transformer_list)
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        return self._combine(Xs)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, name, X, self.transformer_weights)
            for name, trans in self.transformer_list)
        return self._combine(Xs)

    def _combine(self, Xs):
        if all(isinstance(obj, pd.DataFrame) for obj in Xs):
            dfs = []
            for X, transformer in zip(Xs, self.transformer_list):
                df = X.copy()
                df.columns = [transformer[0] + '_' + col for col in X.columns]
                dfs.append(df)
            Xs = dfs[0].join(dfs[1:])
        elif any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs
