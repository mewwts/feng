import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import FeatureUnion as _FeatureUnion, _fit_transform_one, _transform_one
from sklearn.externals.joblib import Parallel, delayed


class FeatureUnion(_FeatureUnion):

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, name, weight, X, y,
                                        **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        return self._combine(Xs)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, name, weight, X)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        return self._combine(Xs)

    def _combine(self, Xs):
        if all(isinstance(obj, pd.DataFrame) for obj in Xs):
            dfs = []
            for X, transformer in zip(Xs, self.transformer_list):
                df = X.copy()
                df.columns = [transformer[0] + '_' + col for col in X.columns]
                dfs.append(df)
            Xs = pd.concat(dfs, axis=1)
        elif any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs
